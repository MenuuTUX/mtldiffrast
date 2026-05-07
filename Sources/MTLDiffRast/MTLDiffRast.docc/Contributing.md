# Contributing Guidelines

Thank you for your interest in contributing to MTLDiffRast! This document provides guidelines and instructions for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Reporting Issues](#reporting-issues)

---

## Code of Conduct

### Our Pledge

We pledge to make participation in MTLDiffRast a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

---

## Getting Started

### Prerequisites

- macOS 12.0+ on Apple Silicon (M1/M2/M3)
- Xcode 15.0+
- Swift 5.9+
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/mtldiffrast.git
   cd mtldiffrast
   ```

3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/mtldiffrast.git
   ```

4. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### Build from Source

```bash
swift build
```

### Run Tests

```bash
swift test
```

---

## Development Workflow

### Branch Naming

Use descriptive branch names:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements
- `test/description` - Test additions/updates

### Commit Messages

Follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(rasterizer): add support for custom depth range

fix(shaders): correct barycentric coordinate calculation

docs(api): update Rasterizer documentation with examples
```

### Keeping Up to Date

Regularly sync with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

---

## Coding Standards

### Swift Style Guide

#### General

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use trailing closures
- Prefer `let` over `var`

#### Naming Conventions

```swift
// Types: PascalCase
public struct RasterOutput { }
public enum RasterizerError { }
public final class Rasterizer { }

// Variables and functions: camelCase
private let commandQueue: MTLCommandQueue
public func rasterize(positions: [SIMD4<Float>]) throws -> RasterOutput

// Constants: camelCase with descriptive names
public let version = "1.0.0"
constant float EPSILON = 1e-8f;

// Protocol names: Describe capability
public protocol Rasterizable { }
```

#### Error Handling

```swift
// Define comprehensive error types
public enum RasterizerError: LocalizedError {
    case metalUnavailable
    case invalidResolution(width: Int, height: Int)
    
    public var errorDescription: String? {
        switch self {
        case .metalUnavailable:
            return "Metal is not available"
        case .invalidResolution(let w, let h):
            return "Invalid resolution: \(w)x\(h)"
        }
    }
}

// Use do-catch with specific error handling
do {
    let output = try rasterizer.rasterize(...)
} catch RasterizerError.metalUnavailable {
    // Handle specific error
} catch {
    // Handle general errors
}
```

#### Documentation Comments

```swift
/// Performs forward pass triangle rasterization.
///
/// This method renders triangles to a depth buffer using GPU acceleration.
/// Each pixel contains the ID of the closest triangle and its depth value.
///
/// - Parameters:
///   - positions: Vertex positions in clip space (x, y, z, w)
///   - triangles: Triangle definitions as vertex indices
///   - width: Output image width in pixels
///   - height: Output image height in pixels
/// - Returns: RasterOutput containing triangle IDs and depth buffer
/// - Throws: RasterizerError if rasterization fails
public func rasterize(
    positions: [SIMD4<Float>],
    triangles: [SIMD3<Int32>],
    width: Int,
    height: Int
) throws -> RasterOutput
```

### Metal Shader Style

```metal
// Use descriptive variable names
kernel void rasterizeKernel(
    device const float4* positions [[buffer(0)]],
    device const int3* triangles [[buffer(1)]],
    device RasterOutput* outputBuffer [[buffer(2)]],
    constant int& triangleCount [[buffer(3)]],
    constant int& width [[buffer(4)]],
    constant int& height [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Early exit for out-of-bounds threads
    if (gid.x >= width || gid.y >= height) {
        return;
    }
    
    // Clear variable naming
    int pixelIndex = gid.y * width + gid.x;
    
    // ... implementation
}
```

---

## Testing

### Writing Tests

All new features must include tests. Tests go in `Tests/MTLDiffRastTests/`.

```swift
import XCTest
@testable import MTLDiffRast

final class RasterizerTests: XCTestCase {
    
    var rasterizer: Rasterizer!
    
    override func setUp() throws {
        guard isMetalAvailable() else {
            throw SkipTest("Metal not available")
        }
        rasterizer = try Rasterizer()
    }
    
    func testSingleTriangleRasterization() throws {
        // Arrange
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(0.0, 0.5, 0.5, 1.0),
            SIMD4<Float>(-0.5, -0.5, 0.5, 1.0),
            SIMD4<Float>(0.5, -0.5, 0.5, 1.0)
        ]
        let triangles: [SIMD3<Int32>] = [SIMD3<Int32>(0, 1, 2)]
        
        // Act
        let output = try rasterizer.rasterize(
            positions: positions,
            triangles: triangles,
            width: 64,
            height: 64
        )
        
        // Assert
        XCTAssertGreaterThan(output.pixelCount, 0)
        XCTAssertTrue(output.triangleIds.contains(where: { $0 >= 0 }))
    }
    
    func testInvalidResolution() throws {
        // Arrange
        let positions: [SIMD4<Float>] = [...]
        let triangles: [SIMD3<Int32>] = [...]
        
        // Act & Assert
        XCTAssertThrowsError(try rasterizer.rasterize(
            positions: positions,
            triangles: triangles,
            width: 0,  // Invalid
            height: 64
        )) { error in
            guard case RasterizerError.invalidResolution = error else {
                XCTFail("Wrong error type")
                return
            }
        }
    }
}
```

### Running Tests

```bash
# All tests
swift test

# Specific test
swift test --filter RasterizerTests.testSingleTriangleRasterization

# With code coverage
swift test --enable-code-coverage
```

### Test Coverage

Aim for >80% code coverage on new features. Check coverage:

```bash
xccov view --report ./build/logs/test.xcresult
```

---

## Documentation

### Updating Documentation

All public APIs must be documented:

1. **Inline documentation**: Use Swift doc comments (`///`)
2. **API Reference**: Update API documentation for significant changes
3. **Examples**: Add examples
4. **FAQ**: Add common questions

### Documentation Style

- Use clear, concise language
- Include code examples
- Explain the "why" not just the "what"
- Link to related documentation

---

## Pull Request Process

### Before Submitting

1. **Rebase on main**: Ensure your branch is up to date
   ```bash
   git rebase upstream/main
   ```

2. **Run all tests**: Ensure everything passes
   ```bash
   swift test
   ```

3. **Check code style**: Ensure consistent formatting
   ```bash
   swift-format lint Sources/
   ```

4. **Update documentation**: Add/update docs as needed

5. **Squash commits**: Combine related commits
   ```bash
   git rebase -i HEAD~N  # N = number of commits
   ```

### Creating PR

1. Push your branch:
   ```bash
   git push origin feature/your-feature
   ```

2. Open pull request on GitHub

3. Fill out PR template:
   - Description of changes
   - Related issues
   - Testing performed
   - Screenshots (if applicable)

### Review Process

1. Maintainers will review within 1 week
2. Address feedback by pushing new commits
3. Once approved, PR will be merged

---

## Reporting Issues

### Bug Reports

Include:

1. **System information**:
   - macOS version
   - Chip type (M1/M2/M3)
   - MTLDiffRast version

2. **Steps to reproduce**:
   ```swift
   // Minimal code example
   let rasterizer = try Rasterizer()
   // ... steps that cause issue
   ```

3. **Expected behavior**: What should happen

4. **Actual behavior**: What actually happens

5. **Error messages**: Full stack trace if available

### Feature Requests

Include:

1. **Problem statement**: What problem does this solve?

2. **Proposed solution**: How should it work?

3. **Use cases**: Example scenarios

4. **Alternatives considered**: Other approaches you've thought about

---

## Release Process

### Version Numbering

MTLDiffRast follows Semantic Versioning (SemVer):

- **MAJOR.MINOR.PATCH** (e.g., 1.2.3)
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Checklist

For maintainers:

- [ ] Update version in `MTLDiffRast.swift`
- [ ] Update CHANGELOG.md
- [ ] Run all tests
- [ ] Create release tag
- [ ] Publish to GitHub Releases
- [ ] Update Swift Package registry
