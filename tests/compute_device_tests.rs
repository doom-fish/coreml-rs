use coreml::prelude::*;

fn assert_supported_or_unavailable(result: Result<Vec<ComputeDevice>, CoreMLError>) {
    match result {
        Ok(devices) => {
            for device in devices {
                match device.kind {
                    ComputeDeviceKind::Cpu
                    | ComputeDeviceKind::Gpu
                    | ComputeDeviceKind::NeuralEngine
                    | ComputeDeviceKind::Unknown => {}
                }
                assert!(!device.description.is_empty());
            }
        }
        Err(CoreMLError::Unsupported(_)) => {}
        Err(error) => panic!("unexpected compute-device discovery error: {error}"),
    }
}

#[test]
fn compute_device_queries_are_callable() {
    assert_supported_or_unavailable(all_compute_devices());
    assert_supported_or_unavailable(Model::available_compute_devices());
}
