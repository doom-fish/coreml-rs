import CoreML
import Foundation

public typealias CMModelCompileAsyncCallback = @convention(c) (
    Int32,
    UnsafeMutablePointer<CChar>?,
    UnsafePointer<CChar>?,
    UnsafeMutableRawPointer?
) -> Void

final class CMModelCompileAsyncCallbackBox: @unchecked Sendable {
    let callback: CMModelCompileAsyncCallback
    let refcon: UnsafeMutableRawPointer?

    init(callback: @escaping CMModelCompileAsyncCallback, refcon: UnsafeMutableRawPointer?) {
        self.callback = callback
        self.refcon = refcon
    }

    func succeed(_ path: String) {
        callback(CM_OK, cm_string(path), nil, refcon)
    }

    func fail(status: Int32, message: String) {
        message.withCString { callback(status, nil, $0, refcon) }
    }

    func fail(error: Error, fallback: Int32) {
        fail(status: cm_status_code(for: error, fallback: fallback), message: error.localizedDescription)
    }
}

@_cdecl("cm_model_compile")
public func cm_model_compile(
    _ pathPtr: UnsafePointer<CChar>?,
    _ outCompiledPath: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outCompiledPath.pointee = nil
    guard let pathPtr else {
        cm_write_error(errorOut, "source model path must not be null")
        return CM_INVALID_ARGUMENT
    }

    let sourceURL = cm_url(from: pathPtr)
    switch cm_block_on_async(work: { try await MLModel.compileModel(at: sourceURL) }) {
    case let .success(compiledURL):
        outCompiledPath.pointee = cm_string(compiledURL.path)
        return CM_OK
    case let .failure(error):
        cm_write_error(errorOut, error.localizedDescription)
        return cm_status_code(for: error, fallback: CM_COMPILATION_FAILED)
    }
}

@_cdecl("cm_model_compile_async")
public func cm_model_compile_async(
    _ pathPtr: UnsafePointer<CChar>?,
    _ callback: @escaping CMModelCompileAsyncCallback,
    _ refcon: UnsafeMutableRawPointer?
) {
    let box = CMModelCompileAsyncCallbackBox(callback: callback, refcon: refcon)
    guard let pathPtr else {
        box.fail(status: CM_INVALID_ARGUMENT, message: "source model path must not be null")
        return
    }

    let sourceURL = cm_url(from: pathPtr)
    Task {
        do {
            let compiledURL = try await MLModel.compileModel(at: sourceURL)
            box.succeed(compiledURL.path)
        } catch {
            box.fail(error: error, fallback: CM_COMPILATION_FAILED)
        }
    }
}
