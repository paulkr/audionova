// The studio scrolls. It grows a metrics grid, a spectrum plot, a node-by-node
// graph inspector and a transport as the session goes on, and there is no
// height at which pinning it to the viewport is right.

export default function AppLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return <div className="min-h-dvh overflow-y-auto">{children}</div>;
}
