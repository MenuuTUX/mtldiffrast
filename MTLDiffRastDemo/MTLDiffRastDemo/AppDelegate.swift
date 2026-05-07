//
//  AppDelegate.swift
//  MTLDiffRastDemo
//

import SwiftUI

@main
struct MTLDiffRastDemoApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 1000, minHeight: 700)
        }
        .windowResizability(.contentMinSize)
        .defaultSize(width: 1180, height: 780)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
