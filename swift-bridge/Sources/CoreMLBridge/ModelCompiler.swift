import CoreML
import Foundation

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
