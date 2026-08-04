import type { Metadata } from "next";
import { Figtree, Raleway } from "next/font/google";
import "./globals.css";

// Roobert stand-in: geometric-humanist, weights 300/400/600 per the reference system.
const figtree = Figtree({
  variable: "--font-figtree",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
});

const raleway = Raleway({
  variable: "--font-raleway",
  subsets: ["latin"],
  weight: ["400"],
});

export const metadata: Metadata = {
  title: "Audionova — Sound, Unbound",
  description:
    "Drop an audio file and process it. Studio-grade audio intelligence.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${figtree.variable} ${raleway.variable} h-full antialiased`}
    >
      {/* NO overflow rule here. The one-page hero wants the viewport pinned;
          the studio is a long document and must scroll. Those are opposite
          requirements, so each route group states its own in its own layout
          rather than one of them fighting the root. */}
      <body className="bg-canvas">{children}</body>
    </html>
  );
}
