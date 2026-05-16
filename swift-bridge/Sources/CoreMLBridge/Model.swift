import CoreML
import Foundation

func cm_error_message(_ error: Error?) -> String {
    (error as NSError?)?.localizedDescription ?? "unknown CoreML error"
}

func cm_feature_description_object(_ description: MLFeatureDescription) -> [String: Any] {
    var object: [String: Any] = [
        "name": description.name,
        "feature_type": cm_feature_type_name(description.type),
        "optional": description.isOptional,
    ]

    if let multiArrayConstraint = description.multiArrayConstraint {
        object["multi_array_constraint"] = [
            "shape": multiArrayConstraint.shape.map(\.intValue),
            "data_type": cm_multi_array_data_type_name(multiArrayConstraint.dataType),
        ]
    }

    if let imageConstraint = description.imageConstraint {
        object["image_constraint"] = [
            "pixels_wide": imageConstraint.pixelsWide,
            "pixels_high": imageConstraint.pixelsHigh,
            "pixel_format_type": UInt32(imageConstraint.pixelFormatType),
        ]
    }

    return object
}

func cm_model_description_object(_ description: MLModelDescription) -> [String: Any] {
    let inputs = description.inputDescriptionsByName.values
        .map(cm_feature_description_object)
        .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }
    let outputs = description.outputDescriptionsByName.values
        .map(cm_feature_description_object)
        .sorted { ($0["name"] as? String ?? "") < ($1["name"] as? String ?? "") }

    var metadata: [String: Any] = [:]
    for (key, value) in description.metadata {
        metadata[key.rawValue] = cm_json_safe(value)
    }

    return [
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata,
        "is_updatable": description.isUpdatable,
    ]
}

@_cdecl("cm_model_load")
public func cm_model_load(
    _ pathPtr: UnsafePointer<CChar>?,
    _ computeUnits: Int32,
    _ allowLowPrecisionAccumulationOnGPU: Bool,
    _ modelDisplayName: UnsafePointer<CChar>?,
    _ outModel: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outModel.pointee = nil
    guard let pathPtr else {
        cm_write_error(errorOut, "model path must not be null")
        return CM_INVALID_ARGUMENT
    }

    do {
        let configuration = cm_make_configuration(
            computeUnits: computeUnits,
            allowLowPrecisionAccumulationOnGPU: allowLowPrecisionAccumulationOnGPU,
            modelDisplayName: modelDisplayName
        )
        let model = try MLModel(contentsOf: cm_url(from: pathPtr), configuration: configuration)
        outModel.pointee = cm_retain(model)
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return CM_MODEL_LOAD_FAILED
    }
}

@_cdecl("cm_model_load_from_specification")
public func cm_model_load_from_specification(
    _ bytesPtr: UnsafePointer<UInt8>?,
    _ byteCount: Int,
    _ computeUnits: Int32,
    _ allowLowPrecisionAccumulationOnGPU: Bool,
    _ modelDisplayName: UnsafePointer<CChar>?,
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
        let configuration = cm_make_configuration(
            computeUnits: computeUnits,
            allowLowPrecisionAccumulationOnGPU: allowLowPrecisionAccumulationOnGPU,
            modelDisplayName: modelDisplayName
        )

        let status = cm_block_on_async(work: {
            try await MLModel.load(asset: asset, configuration: configuration)
        }, onSuccess: { (model: MLModel) in
            outModel.pointee = cm_retain(model)
        })

        if status != CM_OK {
            cm_write_error(errorOut, "failed to load model from in-memory specification")
            return CM_MODEL_ASSET_FAILED
        }

        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return CM_MODEL_ASSET_FAILED
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
    var compiledURL: URL?

    let status = cm_block_on_async(work: {
        try await MLModel.compileModel(at: sourceURL)
    }, onSuccess: { (url: URL) in
        compiledURL = url
    })

    guard status == CM_OK, let compiledURL else {
        cm_write_error(errorOut, "failed to compile CoreML model")
        return CM_COMPILATION_FAILED
    }

    outCompiledPath.pointee = cm_string(compiledURL.path)
    return CM_OK
}

@_cdecl("cm_model_predict")
public func cm_model_predict(
    _ modelPtr: UnsafeMutableRawPointer?,
    _ inputsPtr: UnsafeMutableRawPointer?,
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
        let output = try model.prediction(from: inputs)
        outProvider.pointee = cm_retain(CMFeatureProviderBox(provider: output))
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return CM_PREDICTION_FAILED
    }
}

@_cdecl("cm_model_predict_batch")
public func cm_model_predict_batch(
    _ modelPtr: UnsafeMutableRawPointer?,
    _ batchPtr: UnsafeMutableRawPointer?,
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
        let output = try model.predictions(fromBatch: batch)
        outBatch.pointee = cm_retain(CMBatchProviderBox(batch: output))
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return CM_PREDICTION_FAILED
    }
}

@_cdecl("cm_model_description_json")
public func cm_model_description_json(_ modelPtr: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<CChar>? {
    guard let modelPtr else {
        return cm_string("{}")
    }
    let model: MLModel = cm_borrow(modelPtr)
    return cm_string(cm_json_string(cm_model_description_object(model.modelDescription)))
}
