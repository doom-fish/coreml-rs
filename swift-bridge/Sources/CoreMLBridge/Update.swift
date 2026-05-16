import CoreML
import Foundation

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

@_cdecl("cm_update_run")
public func cm_update_run(
    _ modelPathPtr: UnsafePointer<CChar>?,
    _ trainingDataPtr: UnsafeMutableRawPointer?,
    _ configurationJson: UnsafePointer<CChar>?,
    _ handlersJson: UnsafePointer<CChar>?,
    _ outResultJson: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>,
    _ errorOut: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    outResultJson.pointee = nil
    guard let modelPathPtr, let trainingDataPtr else {
        cm_write_error(errorOut, "update task requires a model path and training data")
        return CM_INVALID_ARGUMENT
    }

    do {
        let configuration = try cm_make_configuration(from: configurationJson)
        let events = try cm_update_events(from: handlersJson)
        let trainingData: CMBatchProviderBox = cm_borrow(trainingDataPtr)
        var contexts: [[String: Any]] = []
        let semaphore = DispatchSemaphore(value: 0)

        let progressHandlers = MLUpdateProgressHandlers(
            forEvents: events,
            progressHandler: { context in
                contexts.append(cm_update_context_object(context))
            },
            completionHandler: { context in
                contexts.append(cm_update_context_object(context, eventOverride: "completion"))
                semaphore.signal()
            }
        )

        let task = try MLUpdateTask(
            forModelAt: cm_url(from: modelPathPtr),
            trainingData: trainingData,
            configuration: configuration,
            progressHandlers: progressHandlers
        )
        task.resume()

        if semaphore.wait(timeout: .now() + .seconds(60)) == .timedOut {
            task.cancel()
            cm_write_error(errorOut, "timed out waiting for CoreML update task completion")
            return CM_TIMED_OUT
        }

        if task.state == .failed {
            cm_write_error(errorOut, task.error?.localizedDescription ?? "CoreML update task failed")
            return CM_UPDATE_FAILED
        }

        outResultJson.pointee = cm_string(cm_json_string([
            "final_state": cm_task_state_name(task.state),
            "contexts": contexts,
        ]))
        return CM_OK
    } catch {
        cm_write_error(errorOut, error.localizedDescription)
        return cm_status_code(for: error, fallback: CM_UPDATE_FAILED)
    }
}
