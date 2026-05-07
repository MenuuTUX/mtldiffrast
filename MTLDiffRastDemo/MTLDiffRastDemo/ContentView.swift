//
//  ContentView.swift
//  MTLDiffRastDemo
//
//  Main UI for the demo application.
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = DemoViewModel()

    var body: some View {
        HStack(spacing: 0) {
            sidebar
                .frame(minWidth: 280, idealWidth: 300, maxWidth: 320)

            Divider()

            stage
        }
        .frame(minWidth: 900, minHeight: 620)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var sidebar: some View {
        VStack(spacing: 0) {
            titleBar

            Divider()

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    ForEach(DemoFeatureGroup.allCases) { group in
                        FeatureSection(
                            title: group.rawValue,
                            features: group.features,
                            selection: viewModel.selectedFeature,
                            action: viewModel.selectFeature
                        )
                    }
                }
                .padding(14)
            }

            Divider()

            controls
        }
        .background(.bar)
    }

    private var titleBar: some View {
        HStack(spacing: 12) {
            Image(systemName: "triangle.inset.filled")
                .font(.system(size: 22, weight: .semibold))
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(.tint)
                .frame(width: 32, height: 32)

            VStack(alignment: .leading, spacing: 2) {
                Text("MTLDiffRast")
                    .font(.title3.weight(.semibold))
                Text("Metal differentiable rasterization")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 16)
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 8) {
                Image(systemName: "slider.horizontal.3")
                    .foregroundStyle(.secondary)
                    .frame(width: 18)
                Text("Options")
                    .font(.headline)
            }

            if viewModel.selectedFeature.supportsWireframe {
                Toggle(isOn: $viewModel.showWireframe) {
                    Label("Wireframe", systemImage: "line.diagonal")
                }
                .toggleStyle(.switch)
            }

            if viewModel.selectedFeature == .performanceBenchmark {
                Divider()

                HStack {
                    Label("Triangles", systemImage: "number")
                    Spacer()
                    Text("\(viewModel.triangleCount)")
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.secondary)
                }

                Slider(value: Binding(
                    get: { Double(viewModel.triangleCount) },
                    set: { viewModel.triangleCount = Int($0) }
                ), in: 1...1000, step: 1)

                Button(action: viewModel.runBenchmark) {
                    Label("Run Benchmark", systemImage: "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)

                if viewModel.renderTime > 0 {
                    Text(String(format: "%.2f ms", viewModel.renderTime))
                        .font(.system(.body, design: .monospaced))
                        .foregroundStyle(.tint)
                        .frame(maxWidth: .infinity, alignment: .trailing)
                }
            }

            if viewModel.selectedFeature.isOriginalSample {
                Label("Raw sample render", systemImage: "sparkle.magnifyingglass")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(16)
        .frame(minHeight: 142, maxHeight: 260, alignment: .top)
    }

    private var stage: some View {
        VStack(spacing: 0) {
            stageHeader

            MetalView(viewModel: viewModel)
                .frame(minWidth: 520, minHeight: 420)
                .background(Color.black)

            statusBar
        }
        .background(Color(nsColor: .textBackgroundColor))
    }

    private var stageHeader: some View {
        HStack(spacing: 14) {
            Image(systemName: viewModel.selectedFeature.symbolName)
                .font(.system(size: 22, weight: .medium))
                .foregroundStyle(.tint)
                .frame(width: 34, height: 34)

            VStack(alignment: .leading, spacing: 2) {
                Text(viewModel.selectedFeature.rawValue)
                    .font(.title3.weight(.semibold))
                    .lineLimit(1)
                Text(viewModel.selectedFeature.description)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)

            StatusPill(
                icon: viewModel.selectedFeature.isAnimated ? "play.circle" : "pause.circle",
                text: viewModel.selectedFeature.isAnimated ? "Live" : "Static"
            )

            StatusPill(
                icon: "cpu",
                text: viewModel.selectedFeature.isOriginalSample ? "Raw" : "Metal"
            )
        }
        .padding(.horizontal, 18)
        .padding(.vertical, 13)
        .background(.regularMaterial)
    }

    private var statusBar: some View {
        HStack(spacing: 18) {
            MetricLabel(icon: "rectangle.dashed", title: "Render", value: "\(viewModel.renderWidth)x\(viewModel.renderHeight)")

            MetricLabel(
                icon: "speedometer",
                title: "FPS",
                value: viewModel.selectedFeature.isAnimated ? String(format: "%.0f", viewModel.displayedFPS) : "idle"
            )

            MetricLabel(icon: "display", title: "GPU", value: viewModel.deviceName)

            Spacer(minLength: 0)

            if viewModel.renderTime > 0, viewModel.selectedFeature == .performanceBenchmark {
                MetricLabel(
                    icon: "timer",
                    title: "Last",
                    value: String(format: "%.2f ms", viewModel.renderTime)
                )
            }
        }
        .font(.caption)
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.bar)
    }
}

private struct FeatureSection: View {
    let title: String
    let features: [DemoFeature]
    let selection: DemoFeature
    let action: (DemoFeature) -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .textCase(.uppercase)
                .padding(.horizontal, 8)

            VStack(spacing: 4) {
                ForEach(features) { feature in
                    FeatureRow(
                        feature: feature,
                        isSelected: selection == feature
                    ) {
                        action(feature)
                    }
                }
            }
        }
    }
}

private struct FeatureRow: View {
    let feature: DemoFeature
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                Image(systemName: feature.symbolName)
                    .font(.system(size: 16, weight: .medium))
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary)
                    .frame(width: 24, height: 24)

                VStack(alignment: .leading, spacing: 2) {
                    Text(feature.rawValue)
                        .font(.callout.weight(isSelected ? .semibold : .regular))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    Text(feature.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
            .background {
                if isSelected {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color.accentColor.opacity(0.16))
                }
            }
        }
        .buttonStyle(.plain)
        .focusable(false)
    }
}

private struct StatusPill: View {
    let icon: String
    let text: String

    var body: some View {
        Label(text, systemImage: icon)
            .font(.caption.weight(.medium))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 9)
            .padding(.vertical, 5)
            .background(.quaternary, in: Capsule())
    }
}

private struct MetricLabel: View {
    let icon: String
    let title: String
    let value: String

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .foregroundStyle(.secondary)
            Text(title)
                .foregroundStyle(.secondary)
            Text(value)
                .fontWeight(.medium)
                .lineLimit(1)
        }
    }
}

#Preview {
    ContentView()
        .frame(width: 1200, height: 800)
}
