import Foundation
import IOBluetooth

// Line protocol on stdio, so the bridge can be driven by hand with `echo` when
// something misbehaves:
//   stdin   <hex>\n         bytes to write on the RFCOMM channel
//   stdout  READY <mtu>     channel open
//           OK <n>          n bytes written
//           RX <hex>        bytes received from the device
//           ERR <msg>
//           CLOSED

func emit(_ s: String) {
    print(s)
    fflush(stdout)
}

func note(_ s: String) {
    FileHandle.standardError.write(Data((s + "\n").utf8))
}

func hexDigit(_ c: UInt8) -> UInt8? {
    switch c {
    case 0x30...0x39: return c - 0x30
    case 0x61...0x66: return c - 0x61 + 10
    case 0x41...0x46: return c - 0x41 + 10
    default: return nil
    }
}

func unhex(_ s: String) -> [UInt8]? {
    let chars = Array(s.utf8)
    guard chars.count % 2 == 0 else { return nil }
    var out = [UInt8]()
    out.reserveCapacity(chars.count / 2)
    var i = 0
    while i < chars.count {
        guard let hi = hexDigit(chars[i]), let lo = hexDigit(chars[i + 1]) else { return nil }
        out.append(hi << 4 | lo)
        i += 2
    }
    return out
}

func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
    bytes.map { String(format: "%02x", $0) }.joined()
}

func describe(_ status: IOReturn) -> String {
    let names: [IOReturn: String] = [
        kIOReturnTimeout: "timeout — the speaker is not answering; wake it or check it is on",
        kIOReturnBusy: "busy — something else holds the channel",
        kIOReturnNoDevice: "no device — is it still paired?",
        kIOReturnNotOpen: "not open",
        kIOReturnNotPermitted: "not permitted — allow Bluetooth for this terminal",
        kIOReturnExclusiveAccess: "exclusive access — another app owns it",
    ]
    return names[status].map { "\($0) (\(status))" } ?? "IOReturn \(status)"
}

final class ChannelDelegate: NSObject, IOBluetoothRFCOMMChannelDelegate {
    func rfcommChannelData(_ channel: IOBluetoothRFCOMMChannel!,
                           data pointer: UnsafeMutableRawPointer!,
                           length: Int) {
        let buf = UnsafeRawBufferPointer(start: pointer, count: length)
        emit("RX " + hex(buf))
    }

    func rfcommChannelClosed(_ channel: IOBluetoothRFCOMMChannel!) {
        emit("CLOSED")
        exit(0)
    }
}

final class SDPDelegate: NSObject {
    var done = false

    @objc func sdpQueryComplete(_ device: IOBluetoothDevice!, status: IOReturn) {
        if status != kIOReturnSuccess {
            emit("ERR sdp query failed \(status)")
        }
        done = true
    }
}

func listPaired() {
    guard let devices = IOBluetoothDevice.pairedDevices() as? [IOBluetoothDevice] else {
        emit("ERR no paired devices")
        exit(1)
    }
    for d in devices {
        emit("\(d.addressString ?? "?")\t\(d.isConnected() ? "connected" : "offline")\t\(d.name ?? "?")")
    }
}

func listServices(_ address: String) {
    guard let device = IOBluetoothDevice(addressString: address) else {
        emit("ERR unknown address \(address)")
        exit(1)
    }

    let delegate = SDPDelegate()
    if device.performSDPQuery(delegate) != kIOReturnSuccess {
        emit("ERR sdp query could not start")
        exit(1)
    }
    let deadline = Date().addingTimeInterval(15)
    while !delegate.done && Date() < deadline {
        RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.1))
    }

    guard let records = device.services as? [IOBluetoothSDPServiceRecord] else {
        emit("ERR no service records")
        exit(1)
    }
    for record in records {
        var channelID: BluetoothRFCOMMChannelID = 0
        let name = record.getServiceName() ?? "?"
        if record.getRFCOMMChannelID(&channelID) == kIOReturnSuccess {
            emit("rfcomm\t\(channelID)\t\(name)")
        } else {
            emit("other\t-\t\(name)")
        }
    }
}

func write(_ channel: IOBluetoothRFCOMMChannel, _ bytes: [UInt8]) {
    // The channel MTU is well under a full image payload, so every write has to
    // be sliced or the device silently drops the tail.
    let mtu = Int(channel.getMTU())
    let total = bytes.count
    var buffer = bytes
    var offset = 0
    buffer.withUnsafeMutableBytes { raw in
        while offset < total {
            let n = min(mtu, total - offset)
            let rc = channel.writeSync(raw.baseAddress!.advanced(by: offset), length: UInt16(n))
            if rc != kIOReturnSuccess {
                emit("ERR write \(rc)")
                return
            }
            offset += n
        }
    }
    if offset == total {
        emit("OK \(total)")
    }
}

func connect(_ address: String, _ channelID: BluetoothRFCOMMChannelID, kickAudio: Bool) {
    guard let device = IOBluetoothDevice(addressString: address) else {
        emit("ERR unknown address \(address)")
        exit(1)
    }

    if kickAudio && device.isConnected() {
        // macOS grabs the speaker as an audio device and that connection owns the
        // RFCOMM channel we need; drop the whole ACL link first.
        note("closing existing connection to release the audio profile")
        device.closeConnection()
        Thread.sleep(forTimeInterval: 1.5)
    }

    var channel: IOBluetoothRFCOMMChannel?
    let delegate = ChannelDelegate()
    var status: IOReturn = kIOReturnError

    // The speaker drops the ACL link when it has been idle, and opening an
    // RFCOMM channel on a dormant device fails rather than waking it.
    for attempt in 1...4 {
        if !device.isConnected() {
            let opened = device.openConnection()
            if opened != kIOReturnSuccess {
                note("attempt \(attempt): baseband connect failed — \(describe(opened))")
                Thread.sleep(forTimeInterval: 2)
                continue
            }
        }
        status = device.openRFCOMMChannelSync(&channel, withChannelID: channelID, delegate: delegate)
        if status == kIOReturnSuccess && channel != nil { break }
        note("attempt \(attempt): RFCOMM open failed — \(describe(status))")
        device.closeConnection()
        Thread.sleep(forTimeInterval: 2)
    }

    guard status == kIOReturnSuccess, let open = channel else {
        emit("ERR open channel \(channelID): \(describe(status))")
        exit(1)
    }

    emit("READY \(open.getMTU())")

    let reader = Thread {
        while let line = readLine(strippingNewline: true) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty { continue }
            guard let bytes = unhex(trimmed) else {
                emit("ERR bad hex")
                continue
            }
            DispatchQueue.main.sync { write(open, bytes) }
        }
        DispatchQueue.main.async {
            open.close()
            exit(0)
        }
    }
    reader.start()

    RunLoop.main.run()
}

let args = CommandLine.arguments
switch args.dropFirst().first {
case "list":
    listPaired()
case "services":
    guard args.count >= 3 else { note("usage: services <address>"); exit(2) }
    listServices(args[2])
case "connect":
    guard args.count >= 3 else { note("usage: connect <address> [channel] [--keep-audio]"); exit(2) }
    let channelID = BluetoothRFCOMMChannelID(args.count >= 4 && !args[3].hasPrefix("--") ? UInt8(args[3]) ?? 1 : 1)
    connect(args[2], channelID, kickAudio: !args.contains("--keep-audio"))
default:
    note("usage: DivoomBridge list | services <address> | connect <address> [channel] [--keep-audio]")
    exit(2)
}
