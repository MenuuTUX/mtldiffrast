//
//  RasterizerError.swift
//  MTLDiffRast
//
//  Structured error type for all GPU resource and validation failures.
//

import Foundation

/// Errors thrown by ``Rasterizer`` and module-level convenience functions.
///
/// All cases conform to `LocalizedError` so they produce human-readable
/// messages through `error.localizedDescription`.
public enum RasterizerError: LocalizedError {

    /// No Metal device could be found on the current host.
    ///
    /// This typically occurs inside a virtual machine or on a Mac with no GPU
    /// that supports Metal.
    case metalUnavailable

    /// A Metal device exists but `makeCommandQueue()` returned `nil`.
    ///
    /// This is rare and usually indicates a driver or resource-exhaustion issue.
    case deviceNotFound

    /// A Metal compute pipeline state could not be created.
    ///
    /// - Parameter reason: A string describing the failure, forwarded from the
    ///   Metal API or the shader compiler.
    case pipelineCreationFailed(String)

    /// The triangle array passed to a rasterize call is empty or negative.
    ///
    /// - Parameter count: The invalid triangle count that was supplied.
    case invalidTriangleCount(Int)

    /// Fewer than three vertices were supplied, or the vertex count doesn't
    /// match the attribute array's implied vertex dimension.
    ///
    /// - Parameter count: The invalid vertex count that was supplied.
    case invalidVertexCount(Int)

    /// The output resolution has a zero or negative dimension.
    ///
    /// - Parameters:
    ///   - width:  The invalid width that was supplied.
    ///   - height: The invalid height that was supplied.
    case invalidResolution(width: Int, height: Int)

    /// `MTLDevice.makeBuffer` returned `nil`.
    ///
    /// - Parameter reason: A label identifying which buffer could not be
    ///   allocated, e.g. `"positions"` or `"gradPos"`.
    case bufferCreationFailed(String)

    /// A command-encoder could not be created or an argument is mismatched.
    ///
    /// - Parameter reason: A description of what could not be encoded.
    case encodingFailed(String)

    /// The GPU command buffer reported an error after `waitUntilCompleted`.
    ///
    /// - Parameter reason: The error description returned by the Metal runtime.
    case commandExecutionFailed(String)

    /// An operation is not available on the current device or configuration.
    ///
    /// - Parameter feature: The name of the unsupported feature.
    case unsupportedFeature(String)

    // MARK: LocalizedError

    public var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            return "Metal is not available on this device."
        case .deviceNotFound:
            return "No Metal device found."
        case .pipelineCreationFailed(let reason):
            return "Failed to create pipeline state: \(reason)"
        case .invalidTriangleCount(let count):
            return "Invalid triangle count: \(count). Must be greater than 0."
        case .invalidVertexCount(let count):
            return "Invalid vertex count: \(count). Must be at least 3."
        case .invalidResolution(let width, let height):
            return "Invalid resolution: \(width)×\(height). Both dimensions must be > 0."
        case .bufferCreationFailed(let reason):
            return "Failed to create Metal buffer: \(reason)"
        case .encodingFailed(let reason):
            return "Failed to encode command: \(reason)"
        case .commandExecutionFailed(let reason):
            return "Command execution failed: \(reason)"
        case .unsupportedFeature(let feature):
            return "Unsupported feature: \(feature)"
        }
    }
}
