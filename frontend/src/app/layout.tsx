import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "OncoVision AI — Histopathological Staging Platform",
  description:
    "Automated cancer cell detection and histopathological staging powered by multimodal LLM inference via Gemini 2.5 Flash.",
  keywords: [
    "cancer detection",
    "histopathology",
    "AI diagnostics",
    "biopsy analysis",
    "oncology",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-[#fafafa] text-black">
        {children}
      </body>
    </html>
  );
}
