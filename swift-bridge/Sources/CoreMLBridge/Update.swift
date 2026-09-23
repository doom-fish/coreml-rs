import CoreML
import Foundation

public typealias CMUpdateEventCallback = @convention(c) (
    UnsafeMutableRawPointer?,
    Int32,
    UnsafePointer<CChar>?,
    UnsafeMutableRawPointer?
) -> Void

public typealias CMContextReleaseCallback = @convention(c) (UnsafeMutableRawPointer?) -> Void

let CM_UPDATE_EVENT_PROGRESS: Int32 = 0
let CM_UPDATE_EVENT_COMPLETION: Int32 = 1

func cm_update_events(from jsonPtr: UnsafePointer<CChar>?) throws -> MLUpdateProgressEvent {
    let object = try cm_json_object(from: jsonPtr)
    let names = object["interested_events"] as? [String] ?? []
    var events: MLUpdateProgressEvent = []
    for name in names {
        switch name {
        case "training_begin":
            events.insert(.trainingBegin)
        case "epoch_end":
            events.insert(.epochEnd)
        case "mini_batch_end":
            events.insert(.miniBatchEnd)
        default:
            break
        }
    }
    return events
}

func cm_update_event_name(_ event: MLUpdateProgressEvent) -> String {
    if event.contains(.trainingBegin) {
        return "training_begin"
    }
    if event.contains(.epochEnd) {
        return "epoch_end"
    }
    if event.contains(.miniBatchEnd) {
        return "mini_batch_end"
    }
    return "unknown"
}

func cm_update_context_object(
    _ context: MLUpdateContext,
    eventOverride: String? = nil
) -> [String: Any] {
    var metrics: [String: Any] = [:]
    for (key, value) in context.metrics {
        metrics[key.scope.map { "\(key.name):\($0)" } ?? key.name] = cm_json_safe(value)
    }

    var parameters: [String: Any] = [:]
    for (key, value) in context.parameters {
        parameters[key.scope.map { "\(key.name):\($0)" } ?? key.name] = cm_json_safe(value)
    }

    return [
        "task_identifier": context.task.taskIdentifier,
        "state": cm_task_state_name(context.task.state),
        "event": eventOverride ?? cm_update_event_name(context.event),
        "metrics": metrics,
        "parameters": parameters,
        "error_message": context.task.error?.localizedDescription as Any,
    ]
}

final class CMUpdateSink: @unchecked Sendable {
    private let callback: CMUpdateEventCallback
    private let context: UnsafeMutableRawPointer?
    private let release: CMContextReleaseCallback

    init(
        callback: @escaping CMUpdateEventCallback,
        context: UnsafeMutableRawPointer?,
        release: @escaping CMContextReleaseCallback
    ) {
        self.callback = callback
        self.context = context
        self.release = release
    }

    deinit {
        release(context)
    }

    func deliver(_ kind: Int32, _ object: [String: Any], model: UnsafeMutableRawPointer?) {
        cm_json_string(object).withCString { callback(context, kind, $0, model) }
    }
}

@_cdecl("cm_update_start")
public func cm_update_start(
    _ modelPathPtr: UnsafePointer<CChar>?,
    _ trainingDataPtr: UnsafeMutableRawPointer?,
    _ configurationJson: UnsafePointer<CChar>?,
    _ handlersJson: UnsafePointer<CChar>?,
    _ callback: @escaping CMUpdateEventCallback,
    _ context: UnsafeMutableRawPointer?,
    _ release: @escaping CMContextReleaseCallback,
    _ outTask: UnsafeMutablePointer<UnsafeMutableRawPointer?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outTask.pointee = nil
    let sink = CMUpdateSink(callback: callback, context: context, release: release)
    guard let modelPathPtr, let trainingDataPtr else {
        cm_write_error(errorOut, "update task requires a model path and training data")
        return CM_INVALID_ARGUMENT
    }

    do {
        let configuration = try cm_make_configuration(from: configurationJson)
        let events = try cm_update_events(from: handlersJson)
        let trainingData = (cm_borrow(trainingDataPtr) as CMBatchProviderBox).snapshot()
        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: events,
            progressHandler: { context in
                sink.deliver(CM_UPDATE_EVENT_PROGRESS, cm_update_context_object(context), model: nil)
            },
            completionHandler: { context in
                let model = context.task.error == nil ? cm_retain(context.model) : nil
                sink.deliver(
                    CM_UPDATE_EVENT_COMPLETION,
                    cm_update_context_object(context, eventOverride: "completion"),
                    model: model
                )
            }
        )

        let task = try MLUpdateTask(
            forModelAt: cm_url(from: modelPathPtr),
            trainingData: trainingData,
            configuration: configuration,
            progressHandlers: progressHandlers
        )
        outTask.pointee = cm_retain(task)
        task.resume()
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return cm_status_code(for: error, fallback: CM_UPDATE_FAILED)
    }
}

@_cdecl("cm_update_cancel")
public func cm_update_cancel(_ taskPtr: UnsafeMutableRawPointer?) {
    guard let taskPtr else { return }
    let task: MLUpdateTask = cm_borrow(taskPtr)
    if task.state == .running || task.state == .suspended {
        task.cancel()
    }
}
