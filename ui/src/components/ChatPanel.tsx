type Props = {
  title: string;
  entries: string[];
};

export default function ChatPanel({ title, entries }: Props) {
  return (
    <div className="panel chat">
      <h3>{title}</h3>
      <div className="chat-log">
        {entries.length === 0 && <div className="muted">No messages yet.</div>}
        {entries.map((entry, idx) => (
          <div key={`${title}-${idx}`} className="chat-entry">
            {entry}
          </div>
        ))}
      </div>
    </div>
  );
}
