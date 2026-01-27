import { useMemo } from "react";
import { ScratchpadEntry } from "../types";

type Props = {
  entries: ScratchpadEntry[];
  isRunning: boolean;
};

export default function ScratchpadPanel({ entries, isRunning }: Props) {
  // Group entries by agent
  const groupedEntries = useMemo(() => {
    const groups: Record<string, ScratchpadEntry[]> = {};
    for (const entry of entries) {
      if (!groups[entry.agent_id]) {
        groups[entry.agent_id] = [];
      }
      groups[entry.agent_id].push(entry);
    }
    return groups;
  }, [entries]);

  const agentIds = Object.keys(groupedEntries).sort();

  if (entries.length === 0 && !isRunning) {
    return null; // Don't show panel if no entries and game is done
  }

  return (
    <div className="scratchpad-panel">
      <div className="scratchpad-header">
        <h3>Agent Scratchpads</h3>
        {isRunning && entries.length === 0 && (
          <span className="scratchpad-waiting">Waiting for agent notes...</span>
        )}
      </div>
      <div className="scratchpad-content">
        {agentIds.map((agentId) => {
          const agentEntries = groupedEntries[agentId];
          const isRed = agentId.startsWith("red");
          const isBlue = agentId.startsWith("blue");
          
          return (
            <div key={agentId} className={`scratchpad-agent ${isRed ? "red" : isBlue ? "blue" : ""}`}>
              <div className="scratchpad-agent-header">
                <span className="scratchpad-agent-name">{formatAgentId(agentId)}</span>
                <span className="scratchpad-agent-count">{agentEntries.length} notes</span>
              </div>
              <div className="scratchpad-entries">
                {agentEntries.map((entry, idx) => (
                  <div key={idx} className="scratchpad-entry">
                    <span className="scratchpad-turn">T{entry.turn}</span>
                    <span className="scratchpad-text">{entry.addition}</span>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function formatAgentId(agentId: string): string {
  // Convert "red_cluer" -> "Red Cluer", "blue_guesser_1" -> "Blue Guesser 1"
  return agentId
    .split("_")
    .map((part, i) => {
      if (i === 0) return part.charAt(0).toUpperCase() + part.slice(1);
      if (part === "guesser") return "Guesser";
      if (part === "cluer") return "Cluer";
      return part;
    })
    .join(" ");
}
