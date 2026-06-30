export function AuthError({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div
      role="alert"
      className="mb-4 rounded-[6px] border border-[#FF2D55]/25 bg-[#FF2D55]/[0.06] px-3.5 py-2.5 text-[0.85rem] font-medium text-[#C81F40]"
    >
      {message}
    </div>
  );
}
