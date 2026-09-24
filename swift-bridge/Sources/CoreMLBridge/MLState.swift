import CoreML
import Darwin
import Foundation

public typealias CMStateMultiArrayCallback = @convention(c) (
    UnsafeMutableRawPointer?,
    UnsafeMutableRawPointer?
) -> Void

#if COREML_HAS_MACOS15_SDK
    final class CMStateGate: @unchecked Sendable {
        private let condition = NSCondition()
        private var held = false
        private var owner: pthread_t?
        private var depth = 0
        private var waiters: [CheckedContinuation<Void, Never>] = []

        func withThreadAccess<T>(_ body: () throws -> T) rethrows -> T {
            enter()
            defer { leave() }
            return try body()
        }

        func acquireForTask() async {
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                condition.lock()
                guard held else {
                    held = true
                    owner = nil
                    depth = 1
                    condition.unlock()
                    continuation.resume()
                    return
                }
                waiters.append(continuation)
                condition.unlock()
            }
        }

        func releaseFromTask() {
            condition.lock()
            depth = 0
            handOff()
        }

        private func enter() {
            condition.lock()
            defer { condition.unlock() }
            let current = pthread_self()
            if held, let owner, pthread_equal(owner, current) != 0 {
                depth += 1
                return
            }
            while held {
                condition.wait()
            }
            held = true
            owner = current
            depth = 1
        }

        private func leave() {
            condition.lock()
            depth -= 1
            guard depth == 0 else {
                condition.unlock()
                return
            }
            handOff()
        }

        private func handOff() {
            guard !waiters.isEmpty else {
                held = false
                owner = nil
                condition.broadcast()
                condition.unlock()
                return
            }
            let next = waiters.removeFirst()
            owner = nil
            depth = 1
            condition.unlock()
            next.resume()
        }
    }

    @available(macOS 15.0, *)
    final class CMStateBox: NSObject, @unchecked Sendable {
        let state: MLState
        let stateNames: Set<String>
        let gate = CMStateGate()

        init(state: MLState, stateNames: Set<String>) {
            self.state = state
            self.stateNames = stateNames
            super.init()
        }
    }

    @_cdecl("cm_state_runtime_supported")
    public func cm_state_runtime_supported() -> Bool {
        if #available(macOS 15.0, *) {
            return true
        }
        return false
    }

    @_cdecl("cm_model_new_state")
    public func cm_model_new_state(
        _ modelPtr: UnsafeMutableRawPointer?,
        _ outState: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outState.pointee = nil
        guard let modelPtr else {
            cm_write_error(errorOut, "model must not be null")
            return CM_INVALID_ARGUMENT
        }
        if #available(macOS 15.0, *) {
            let model: MLModel = cm_borrow(modelPtr)
            let stateNames = Set(model.modelDescription.stateDescriptionsByName.keys)
            guard !stateNames.isEmpty else {
                cm_write_error(errorOut, "model declares no state features")
                return CM_STATE_FAILED
            }
            outState.pointee = cm_retain(CMStateBox(state: model.makeState(), stateNames: stateNames))
            return CM_OK
        }
        cm_write_error(errorOut, "MLState requires macOS 15.0+")
        return CM_UNSUPPORTED
    }

    @_cdecl("cm_model_predict_with_state")
    public func cm_model_predict_with_state(
        _ modelPtr: UnsafeMutableRawPointer?,
        _ inputsPtr: UnsafeMutableRawPointer?,
        _ statePtr: UnsafeMutableRawPointer?,
        _ predictionOptionsJson: UnsafePointer<CChar>?,
        _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outProvider.pointee = nil
        guard let modelPtr, let inputsPtr, let statePtr else {
            cm_write_error(errorOut, "model, inputs, and state must not be null")
            return CM_INVALID_ARGUMENT
        }
        if #available(macOS 15.0, *) {
            let model: MLModel = cm_borrow(modelPtr)
            let inputs: CMFeatureProviderBox = cm_borrow(inputsPtr)
            let box: CMStateBox = cm_borrow(statePtr)
            do {
                let options = try cm_make_prediction_options(from: predictionOptionsJson)
                let output = try box.gate.withThreadAccess {
                    try model.prediction(from: inputs, using: box.state, options: options)
                }
                outProvider.pointee = cm_retain(CMFeatureProviderBox(provider: output))
                return CM_OK
            } catch {
                cm_write_error(errorOut, error.localizedDescription)
                return cm_status_code(for: error, fallback: CM_STATE_FAILED)
            }
        }
        cm_write_error(errorOut, "MLState requires macOS 15.0+")
        return CM_UNSUPPORTED
    }

    @_cdecl("cm_model_predict_with_state_async")
    public func cm_model_predict_with_state_async(
        _ modelPtr: UnsafeMutableRawPointer?,
        _ inputsPtr: UnsafeMutableRawPointer?,
        _ statePtr: UnsafeMutableRawPointer?,
        _ predictionOptionsJson: UnsafePointer<CChar>?,
        _ callback: @escaping CMModelAsyncCallback,
        _ refcon: UnsafeMutableRawPointer?
    ) -> UnsafeMutableRawPointer? {
        let callbackBox = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
        guard let modelPtr, let inputsPtr, let statePtr else {
            callbackBox.fail(status: CM_INVALID_ARGUMENT, message: "model, inputs, and state must not be null")
            return nil
        }
        if #available(macOS 15.0, *) {
            let model: MLModel = cm_borrow(modelPtr)
            let inputs = (cm_borrow(inputsPtr) as CMFeatureProviderBox).snapshot()
            let box: CMStateBox = cm_borrow(statePtr)
            do {
                let options = try cm_make_prediction_options(from: predictionOptionsJson)
                let task = Task {
                    await box.gate.acquireForTask()
                    let result: Result<any MLFeatureProvider, Error>
                    do {
                        try Task.checkCancellation()
                        result = .success(try await model.prediction(from: inputs, using: box.state, options: options))
                    } catch {
                        result = .failure(error)
                    }
                    box.gate.releaseFromTask()
                    switch result {
                    case let .success(output):
                        callbackBox.succeed(cm_retain(CMFeatureProviderBox(provider: output)))
                    case let .failure(error):
                        callbackBox.fail(error: error, fallback: CM_STATE_FAILED)
                    }
                }
                return cm_retain(CMTaskHandle(task))
            } catch {
                callbackBox.fail(error: error, fallback: CM_STATE_FAILED)
                return nil
            }
        }
        callbackBox.fail(status: CM_UNSUPPORTED, message: "MLState requires macOS 15.0+")
        return nil
    }

    @_cdecl("cm_state_with_multi_array")
    public func cm_state_with_multi_array(
        _ statePtr: UnsafeMutableRawPointer?,
        _ namePtr: UnsafePointer<CChar>?,
        _ callback: CMStateMultiArrayCallback,
        _ context: UnsafeMutableRawPointer?,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        guard let statePtr, let namePtr else {
            cm_write_error(errorOut, "state and state name must not be null")
            return CM_INVALID_ARGUMENT
        }
        if #available(macOS 15.0, *) {
            let box: CMStateBox = cm_borrow(statePtr)
            let stateName = String(cString: namePtr)
            guard box.stateNames.contains(stateName) else {
                cm_write_error(errorOut, "model declares no state named '\(stateName)'")
                return CM_INVALID_ARGUMENT
            }
            box.gate.withThreadAccess {
                box.state.withMultiArray(for: stateName) { buffer in
                    callback(Unmanaged.passUnretained(buffer).toOpaque(), context)
                }
            }
            return CM_OK
        }
        cm_write_error(errorOut, "MLState requires macOS 15.0+")
        return CM_UNSUPPORTED
    }
#else
    @_cdecl("cm_state_runtime_supported")
    public func cm_state_runtime_supported() -> Bool { false }

    @_cdecl("cm_model_new_state")
    public func cm_model_new_state(
        _: UnsafeMutableRawPointer?,
        _ outState: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outState.pointee = nil
        cm_write_error(errorOut, "MLState requires a macOS 15.0+ SDK")
        return CM_UNSUPPORTED
    }

    @_cdecl("cm_model_predict_with_state")
    public func cm_model_predict_with_state(
        _: UnsafeMutableRawPointer?,
        _: UnsafeMutableRawPointer?,
        _: UnsafeMutableRawPointer?,
        _: UnsafePointer<CChar>?,
        _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outProvider.pointee = nil
        cm_write_error(errorOut, "MLState requires a macOS 15.0+ SDK")
        return CM_UNSUPPORTED
    }

    @_cdecl("cm_model_predict_with_state_async")
    public func cm_model_predict_with_state_async(
        _: UnsafeMutableRawPointer?,
        _: UnsafeMutableRawPointer?,
        _: UnsafeMutableRawPointer?,
        _: UnsafePointer<CChar>?,
        _ callback: @escaping CMModelAsyncCallback,
        _ refcon: UnsafeMutableRawPointer?
    ) -> UnsafeMutableRawPointer? {
        let box = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
        box.fail(status: CM_UNSUPPORTED, message: "MLState requires a macOS 15.0+ SDK")
        return nil
    }

    @_cdecl("cm_state_with_multi_array")
    public func cm_state_with_multi_array(
        _: UnsafeMutableRawPointer?,
        _: UnsafePointer<CChar>?,
        _: CMStateMultiArrayCallback,
        _: UnsafeMutableRawPointer?,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        cm_write_error(errorOut, "MLState requires a macOS 15.0+ SDK")
        return CM_UNSUPPORTED
    }
#endif
