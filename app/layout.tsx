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
      <body className="h-full overflow-hidden bg-canvas">{children}</body>
    </html>
  );
}
