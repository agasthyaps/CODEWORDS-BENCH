import { useEffect, useRef } from "react";

type Props = {
  title: string;
  entries: string[];
  variant?: "codenames" | "decrypto";
  teamColor?: "red" | "blue";  // For Decrypto where panels are already split by team
  thinking?: string | boolean;  // Show thinking indicator with optional custom message
};

function getEntryClass(entry: string, variant: string, teamColor?: string): string {
  const lower = entry.toLowerCase();
  let classes = "chat-entry";
  
  // For Codenames, detect team from message content
  if (variant === "codenames") {
    if (lower.includes("[red]")) {
      classes += " red-team";
    } else if (lower.includes("[blue]")) {
      classes += " blue-team";
    }
    
    if (lower.includes("clue:")) {
      classes += " clue";
    } else if (lower.includes("guess:")) {
      classes += " guess";
    }
  }
  
  // For Decrypto, use the teamColor prop for consistent styling
  if (variant === "decrypto") {
    // Apply team color consistently
    if (teamColor === "red") {
      classes += " red-team";
    } else if (teamColor === "blue") {
      classes += " blue-team";
    }
    
    // Message type styling
    if (lower.startsWith("round ") && lower.includes(":")) {
      classes += " msg-round";
    } else if (lower.includes("consensus")) {
      classes += " msg-consensus";
    } else if (lower.trim().startsWith("share")) {
      classes += " msg-share";
    } else if (lower.startsWith("[dec]")) {
      classes += " msg-decode";
    } else if (lower.startsWith("[int]")) {
      classes += " msg-intercept";
    }
  }
  
  return classes;
}

export default function ChatPanel({ title, entries, variant = "codenames", teamColor, thinking }: Props) {
  const logRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [entries, thinking]);

  const thinkingMessage = typeof thinking === "string" ? thinking : "Thinking";

  return (
    <div className="panel chat">
      <h3>{title}</h3>
      <div className="chat-log" ref={logRef}>
        {entries.length === 0 && !thinking && (
          <div className="empty-state">No messages yet</div>
        )}
        {entries.map((entry, idx) => (
          <div key={`${title}-${idx}`} className={getEntryClass(entry, variant, teamColor)}>
            {entry}
          </div>
        ))}
        {thinking && (
          <div className="thinking-indicator">
            <span className="thinking-dots">
              <span></span>
              <span></span>
              <span></span>
            </span>
            <span className="thinking-text">{thinkingMessage}</span>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
