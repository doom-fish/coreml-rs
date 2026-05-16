use coreml::prelude::*;

fn main() {
    match all_compute_devices() {
        Ok(devices) => println!("all compute devices: {devices:#?}"),
        Err(error) => println!("all compute devices unavailable: {error}"),
    }

    match Model::available_compute_devices() {
        Ok(devices) => println!("model-visible compute devices: {devices:#?}"),
        Err(error) => println!("model-visible compute devices unavailable: {error}"),
    }
}
