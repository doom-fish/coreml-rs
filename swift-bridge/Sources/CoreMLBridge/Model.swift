import CoreML
import Foundation

public typealias CMModelAsyncCallback = @convention(c) (
  Int32,
  UnsafeMutableRawPointer?,
  UnsafePointer<CChar>?,
  UnsafeMutableRawPointer?
) -> Void

final class CMModelAsyncCallbackBox: @unchecked Sendable {
  let callback: CMModelAsyncCallback
  let refcon: UnsafeMutableRawPointer?

  init(callback: @escaping CMModelAsyncCallback, refcon: UnsafeMutableRawPointer?) {
    self.callback = callback
    self.refcon = refcon
  }

  func succeed(_ result: UnsafeMutableRawPointer) {
    callback(CM_OK, result, nil, refcon)
  }

  func fail(status: Int32, message: String) {
    message.withCString { callback(status, nil, $0, refcon) }
  }

  func fail(error: Error, fallback: Int32) {
    fail(status: cm_status_code(for: error, fallback: fallback), message: error.localizedDescription)
  }
}

@_cdecl("cm_model_load")
public func cm_model_load(
  _ pathPtr: UnsafePointer<CChar>?,
  _ configurationJson: UnsafePointer<CChar>?,
  _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outModel.pointee = nil
  guard let pathPtr else {
    cm_write_error(errorOut, "model path must not be null")
    return CM_INVALID_ARGUMENT
  }

  do {
    let configuration = try cm_make_configuration(from: configurationJson)
    let model = try MLModel(contentsOf: cm_url(from: pathPtr), configuration: configuration)
    outModel.pointee = cm_retain(model)
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_MODEL_LOAD_FAILED)
  }
}

@_cdecl("cm_model_load_async")
public func cm_model_load_async(
  _ pathPtr: UnsafePointer<CChar>?,
  _ configurationJson: UnsafePointer<CChar>?,
  _ callback: @escaping CMModelAsyncCallback,
  _ refcon: UnsafeMutableRawPointer?
) {
  let box = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
  guard let pathPtr else {
    box.fail(status: CM_INVALID_ARGUMENT, message: "model path must not be null")
    return
  }

  do {
    let configuration = try cm_make_configuration(from: configurationJson)
    let url = cm_url(from: pathPtr)
    MLModel.load(contentsOf: url, configuration: configuration) { result in
      switch result {
      case let .success(model):
        box.succeed(cm_retain(model))
      case let .failure(error):
        box.fail(error: error, fallback: CM_MODEL_LOAD_FAILED)
      }
    }
  } catch {
    box.fail(error: error, fallback: CM_MODEL_LOAD_FAILED)
  }
}

@_cdecl("cm_model_load_from_specification")
public func cm_model_load_from_specification(
  _ bytesPtr: UnsafePointer<UInt8>?,
  _ byteCount: Int,
  _ configurationJson: UnsafePointer<CChar>?,
  _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outModel.pointee = nil
  guard let bytesPtr else {
    cm_write_error(errorOut, "specification bytes must not be null")
    return CM_INVALID_ARGUMENT
  }

  do {
    let data = Data(bytes: bytesPtr, count: byteCount)
    let asset = try MLModelAsset(specification: data)
    let configuration = try cm_make_configuration(from: configurationJson)
    switch cm_block_on_async(work: {
      try await MLModel.load(asset: asset, configuration: configuration)
    }) {
    case .success(let model):
      outModel.pointee = cm_retain(model)
      return CM_OK
    case .failure(let error):
      cm_write_error(errorOut, error.localizedDescription)
      return cm_status_code(for: error, fallback: CM_MODEL_ASSET_FAILED)
    }
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_MODEL_ASSET_FAILED)
  }
}

@_cdecl("cm_model_predict")
public func cm_model_predict(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputsPtr: UnsafeMutableRawPointer?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  cm_model_predict_with_options(modelPtr, inputsPtr, nil, outProvider, errorOut)
}

@_cdecl("cm_model_predict_async")
public func cm_model_predict_async(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputsPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ callback: @escaping CMModelAsyncCallback,
  _ refcon: UnsafeMutableRawPointer?
) {
  let box = CMModelAsyncCallbackBox(callback: callback, refcon: refcon)
  guard let modelPtr, let inputsPtr else {
    box.fail(status: CM_INVALID_ARGUMENT, message: "model and inputs must not be null")
    return
  }
  guard #available(macOS 14.0, *) else {
    box.fail(status: CM_UNSUPPORTED, message: "asynchronous prediction requires macOS 14.0+")
    return
  }

  let model: MLModel = cm_borrow(modelPtr)
  let inputs: CMFeatureProviderBox = cm_borrow(inputsPtr)

  do {
    let options = try cm_make_prediction_options(from: predictionOptionsJson)
    Task {
      do {
        let output = try await model.prediction(from: inputs, options: options)
        box.succeed(cm_retain(CMFeatureProviderBox(provider: output)))
      } catch {
        box.fail(error: error, fallback: CM_PREDICTION_FAILED)
      }
    }
  } catch {
    box.fail(error: error, fallback: CM_PREDICTION_FAILED)
  }
}

@_cdecl("cm_model_predict_with_options")
public func cm_model_predict_with_options(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ inputsPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outProvider: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outProvider.pointee = nil
  guard let modelPtr, let inputsPtr else {
    cm_write_error(errorOut, "model and inputs must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  let inputs: CMFeatureProviderBox = cm_borrow(inputsPtr)

  do {
    let options = try cm_make_prediction_options(from: predictionOptionsJson)
    let output = try model.prediction(from: inputs, options: options)
    outProvider.pointee = cm_retain(CMFeatureProviderBox(provider: output))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_PREDICTION_FAILED)
  }
}

@_cdecl("cm_model_predict_batch")
public func cm_model_predict_batch(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ batchPtr: UnsafeMutableRawPointer?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  cm_model_predict_batch_with_options(modelPtr, batchPtr, nil, outBatch, errorOut)
}

@_cdecl("cm_model_predict_batch_with_options")
public func cm_model_predict_batch_with_options(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ batchPtr: UnsafeMutableRawPointer?,
  _ predictionOptionsJson: UnsafePointer<CChar>?,
  _ outBatch: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  outBatch.pointee = nil
  guard let modelPtr, let batchPtr else {
    cm_write_error(errorOut, "model and batch inputs must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  let batch: CMBatchProviderBox = cm_borrow(batchPtr)

  do {
    let options = try cm_make_prediction_options(from: predictionOptionsJson)
    let output = try model.predictions(from: batch, options: options)
    outBatch.pointee = cm_retain(CMBatchProviderBox(batch: output))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_PREDICTION_FAILED)
  }
}

@_cdecl("cm_model_write_to_url")
public func cm_model_write_to_url(
  _ modelPtr: UnsafeMutableRawPointer?,
  _ pathPtr: UnsafePointer<CChar>?,
  _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
  guard let modelPtr, let pathPtr else {
    cm_write_error(errorOut, "model and destination path must not be null")
    return CM_INVALID_ARGUMENT
  }

  let model: MLModel = cm_borrow(modelPtr)
  guard let writable = model as? any MLWritable else {
    cm_write_error(errorOut, "loaded model does not conform to MLWritable")
    return CM_UNSUPPORTED
  }

  do {
    try writable.write(to: cm_url(from: pathPtr))
    return CM_OK
  } catch {
    cm_write_error(errorOut, error.localizedDescription)
    return cm_status_code(for: error, fallback: CM_UPDATE_FAILED)
  }
}
