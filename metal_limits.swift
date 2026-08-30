import Foundation
import Metal

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("No default Metal device")
}

let result: [String: Any] = [
    "device": device.name,
    "max_buffer_length_bytes": device.maxBufferLength,
    "recommended_max_working_set_bytes": device.recommendedMaxWorkingSetSize,
]
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
print(String(data: data, encoding: .utf8)!)
