//
//  MTLDiffRast.swift
//  MTLDiffRast
//
//  Module entry point, device queries, and convenience constructors.
//

import Metal
import Foundation

// MARK: - Device availability

/// Returns `true` when a Metal device is available on the current host.
///
/// Call this before allocating a ``Rasterizer`` in environments where GPU
/// availability is uncertain (e.g., virtual machines or Mac Catalyst).
public func isMetalAvailable() -> Bool {
    MTLCreateSystemDefaultDevice() != nil
}

/// Returns `true` when the process is running on Apple Silicon (arm64).
///
/// Apple Silicon unifies CPU and GPU memory, enabling zero-copy buffer
/// sharing between Swift and Metal kernels.  All performance-critical paths
/// in ``Rasterizer`` rely on `storageModeShared` buffers, which require a
/// unified-memory device.
public func isAppleSilicon() -> Bool {
    #if arch(arm64)
    return true
    #else
    return false
    #endif
}

// MARK: - Device info

/// A snapshot of Metal device capabilities relevant to differentiable rendering.
public struct MetalDeviceInfo {
    /// Human-readable device name reported by the GPU driver.
    public let name: String

    /// Unique registry identifier for the Metal device.
    public let registryID: UInt64

    /// `true` if the device is low-power (e.g., an integrated GPU).
    public let isLowPower: Bool

    /// `true` if the device has no display connection.
    public let isHeadless: Bool

    /// Maximum number of threads per threadgroup on this device.
    public let maxThreadsPerThreadgroup: MTLSize

    /// `true` if the device supports Tier 1 or Tier 2 argument buffers.
    public let isArgumentBufferSupported: Bool

    /// Always `true`; rasterization is supported on all Metal devices.
    public let isRasterizationEnabled: Bool

    /// `true` if the device supports Apple GPU family 7 (A15 / M2 and later).
    public let supportsFamilyApple7: Bool
}

/// Returns capability information about the system default Metal device, or
/// `nil` if no Metal device is present.
///
/// - Returns: A ``MetalDeviceInfo`` snapshot, or `nil`.
public func getMetalDeviceInfo() -> MetalDeviceInfo? {
    guard let device = MTLCreateSystemDefaultDevice() else { return nil }

    let argumentBuffersSupported =
        device.argumentBuffersSupport == .tier1 ||
        device.argumentBuffersSupport == .tier2

    return MetalDeviceInfo(
        name: device.name,
        registryID: device.registryID,
        isLowPower: device.isLowPower,
        isHeadless: device.isHeadless,
        maxThreadsPerThreadgroup: device.maxThreadsPerThreadgroup,
        isArgumentBufferSupported: argumentBuffersSupported,
        isRasterizationEnabled: true,
        supportsFamilyApple7: device.supportsFamily(.apple7)
    )
}

// MARK: - Convenience constructor

/// Creates a ``Rasterizer`` backed by the system default Metal device.
///
/// This is a convenience wrapper around ``Rasterizer/init()`` that surfaces
/// a ``RasterizerError/metalUnavailable`` error before attempting GPU
/// resource allocation.
///
/// - Throws: ``RasterizerError/metalUnavailable`` if no Metal device exists;
///   ``RasterizerError/deviceNotFound`` or
///   ``RasterizerError/pipelineCreationFailed(_:)`` for GPU resource failures.
/// - Returns: A fully initialised ``Rasterizer`` ready for use.
public func createRasterizer() throws -> Rasterizer {
    guard isMetalAvailable() else {
        throw RasterizerError.metalUnavailable
    }
    return try Rasterizer()
}

// MARK: - Package metadata

/// The semantic version of the MTLDiffRast package.
public let version = "1.0.0"
