import AVFoundation
import CoreImage
import Foundation

// One JPEG from a camera, so the test loop can look at the LED panel itself
// instead of asking a human what changed.
//   Snapshot list
//   Snapshot <out.jpg> [--device <index>] [--warmup <frames>]

func emit(_ s: String) {
    print(s)
    fflush(stdout)
}

func fail(_ s: String) -> Never {
    FileHandle.standardError.write(Data((s + "\n").utf8))
    exit(1)
}

func cameras() -> [AVCaptureDevice] {
    var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .external]
    if #available(macOS 14.0, *) {
        types.append(.continuityCamera)
    }
    return AVCaptureDevice.DiscoverySession(
        deviceTypes: types, mediaType: .video, position: .unspecified
    ).devices
}

final class Grabber: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    private let url: URL
    private var remaining: Int
    private let context = CIContext()
    private(set) var done = false

    init(url: URL, warmup: Int) {
        self.url = url
        self.remaining = warmup
    }

    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard !done else { return }
        // Early frames come out black or badly exposed while the sensor settles.
        if remaining > 0 {
            remaining -= 1
            return
        }
        guard let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvPixelBuffer: buffer)
        guard let jpeg = context.jpegRepresentation(
            of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]
        ) else { return }
        do {
            try jpeg.write(to: url)
            emit("\(Int(image.extent.width))x\(Int(image.extent.height)) \(url.path)")
            done = true
        } catch {
            fail("write failed: \(error)")
        }
    }
}

func flag(_ name: String, _ fallback: Int) -> Int {
    guard let i = CommandLine.arguments.firstIndex(of: name),
          i + 1 < CommandLine.arguments.count,
          let value = Int(CommandLine.arguments[i + 1]) else { return fallback }
    return value
}

let argv = CommandLine.arguments
guard argv.count >= 2 else {
    fail("usage: Snapshot list | Snapshot <out.jpg> [--device N] [--warmup N]")
}

if argv[1] == "list" {
    for (i, device) in cameras().enumerated() {
        emit("\(i)\t\(device.localizedName)")
    }
    exit(0)
}

let output = URL(fileURLWithPath: argv[1])
let index = flag("--device", 0)
let warmup = flag("--warmup", 30)

let semaphore = DispatchSemaphore(value: 0)
var granted = false
AVCaptureDevice.requestAccess(for: .video) { ok in
    granted = ok
    semaphore.signal()
}
semaphore.wait()
guard granted else { fail("camera access denied — allow it in System Settings > Privacy & Security > Camera") }

let devices = cameras()
guard index < devices.count else { fail("no camera at index \(index); try `Snapshot list`") }
let camera = devices[index]

let session = AVCaptureSession()
session.sessionPreset = .high
guard let input = try? AVCaptureDeviceInput(device: camera), session.canAddInput(input) else {
    fail("cannot open \(camera.localizedName)")
}
session.addInput(input)

let grabber = Grabber(url: output, warmup: warmup)
let videoOutput = AVCaptureVideoDataOutput()
videoOutput.alwaysDiscardsLateVideoFrames = true
videoOutput.setSampleBufferDelegate(grabber, queue: DispatchQueue(label: "snapshot"))
guard session.canAddOutput(videoOutput) else { fail("cannot attach video output") }
session.addOutput(videoOutput)

session.startRunning()
let deadline = Date().addingTimeInterval(20)
while !grabber.done && Date() < deadline {
    RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.05))
}
session.stopRunning()

if !grabber.done {
    fail("timed out waiting for a frame from \(camera.localizedName)")
}
