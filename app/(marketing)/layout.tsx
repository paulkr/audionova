// The one-pager. `h-full overflow-hidden` is what makes the hero a fixed
// viewport with the orb bleeding off every edge — it is load-bearing for the
// design, not a leftover, which is why it lives here rather than at the root
// where it would also pin the studio.

export default function MarketingLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return <div className="h-dvh overflow-hidden">{children}</div>;
}
