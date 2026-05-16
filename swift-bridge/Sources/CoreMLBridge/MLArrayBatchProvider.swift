import Foundation

@_cdecl("cm_batch_provider_new")
public func cm_batch_provider_new() -> UnsafeMutableRawPointer? {
    cm_retain(CMBatchProviderBox())
}

@_cdecl("cm_batch_provider_push")
public func cm_batch_provider_push(
    _ batchPtr: UnsafeMutableRawPointer?,
    _ providerPtr: UnsafeMutableRawPointer?
) -> Int32 {
    guard let batchPtr, let providerPtr else {
        return CM_INVALID_ARGUMENT
    }

    let batch: CMBatchProviderBox = cm_borrow(batchPtr)
    let provider: CMFeatureProviderBox = cm_borrow(providerPtr)
    batch.items.append(CMFeatureProviderBox(provider: provider))
    return CM_OK
}
