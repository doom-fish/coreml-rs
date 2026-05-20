import CoreML
import Foundation

#if COREML_HAS_MACOS15_SDK
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
            guard let unmanagedState = model.perform(NSSelectorFromString("newState")),
                  let state = unmanagedState.takeUnretainedValue() as? MLState else {
                cm_write_error(errorOut, "model did not vend an MLState instance")
                return CM_STATE_FAILED
            }
            outState.pointee = cm_retain(state)
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
            let state: MLState = cm_borrow(statePtr)
            do {
                let options = try cm_make_prediction_options(from: predictionOptionsJson)
                let output = try model.prediction(from: inputs, using: state, options: options)
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
    ) {
        let box = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
        guard let modelPtr, let inputsPtr, let statePtr else {
            box.fail(status: CM_INVALID_ARGUMENT, message: "model, inputs, and state must not be null")
            return
        }
        if #available(macOS 15.0, *) {
            let model: MLModel = cm_borrow(modelPtr)
            let inputs: CMFeatureProviderBox = cm_borrow(inputsPtr)
            let state: MLState = cm_borrow(statePtr)
            do {
                let options = try cm_make_prediction_options(from: predictionOptionsJson)
                Task {
                    do {
                        let output = try await model.prediction(from: inputs, using: state, options: options)
                        box.succeed(cm_retain(CMFeatureProviderBox(provider: output)))
                    } catch {
                        box.fail(error: error, fallback: CM_STATE_FAILED)
                    }
                }
            } catch {
                box.fail(error: error, fallback: CM_STATE_FAILED)
            }
            return
        }
        box.fail(status: CM_UNSUPPORTED, message: "MLState requires macOS 15.0+")
    }

    @_cdecl("cm_state_snapshot_multi_array")
    public func cm_state_snapshot_multi_array(
        _ statePtr: UnsafeMutableRawPointer?,
        _ namePtr: UnsafePointer<CChar>?,
        _ outArray: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outArray.pointee = nil
        guard let statePtr, let namePtr else {
            cm_write_error(errorOut, "state and state name must not be null")
            return CM_INVALID_ARGUMENT
        }
        if #available(macOS 15.0, *) {
            let state: MLState = cm_borrow(statePtr)
            let stateName = String(cString: namePtr)
            var copiedArray: MLMultiArray?
            var capturedError: Error?
            state.withMultiArray(for: stateName) { buffer in
                do {
                    copiedArray = try cm_copy_multi_array(buffer)
                } catch {
                    capturedError = error
                }
            }
            if let capturedError {
                cm_write_error(errorOut, capturedError.localizedDescription)
                return cm_status_code(for: capturedError, fallback: CM_STATE_FAILED)
            }
            guard let copiedArray else {
                cm_write_error(errorOut, "state buffer '\(stateName)' was unavailable")
                return CM_STATE_FAILED
            }
            outArray.pointee = cm_retain(copiedArray)
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
    ) {
        let box = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
        box.fail(status: CM_UNSUPPORTED, message: "MLState requires a macOS 15.0+ SDK")
    }

    @_cdecl("cm_state_snapshot_multi_array")
    public func cm_state_snapshot_multi_array(
        _: UnsafeMutableRawPointer?,
        _: UnsafePointer<CChar>?,
        _ outArray: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
        _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
    ) -> Int32 {
        outArray.pointee = nil
        cm_write_error(errorOut, "MLState requires a macOS 15.0+ SDK")
        return CM_UNSUPPORTED
    }
#endif
