import CoreML
import Foundation

final class CMBatchProviderBox: NSObject, MLBatchProvider {
    var items: [CMFeatureProviderBox]

    init(items: [CMFeatureProviderBox] = []) {
        self.items = items
    }

    convenience init(batch: any MLBatchProvider) {
        var items: [CMFeatureProviderBox] = []
        items.reserveCapacity(batch.count)
        for index in 0..<batch.count {
            items.append(CMFeatureProviderBox(provider: batch.features(at: index)))
        }
        self.init(items: items)
    }

    var count: Int {
        items.count
    }

    func features(at index: Int) -> any MLFeatureProvider {
        items[index]
    }
}

@_cdecl("cm_batch_provider_count")
public func cm_batch_provider_count(_ batchPtr: UnsafeMutableRawPointer?) -> Int {
    guard let batchPtr else { return 0 }
    let batch: CMBatchProviderBox = cm_borrow(batchPtr)
    return batch.count
}

@_cdecl("cm_batch_provider_get_provider")
public func cm_batch_provider_get_provider(
    _ batchPtr: UnsafeMutableRawPointer?,
    _ index: Int
) -> UnsafeMutableRawPointer? {
    guard let batchPtr else { return nil }
    let batch: CMBatchProviderBox = cm_borrow(batchPtr)
    guard index >= 0 && index < batch.count else {
        return nil
    }
    return cm_retain(batch.items[index])
}
