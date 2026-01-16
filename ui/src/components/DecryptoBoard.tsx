type CurrentRoundInfo = {
  code: number[];
  clues: string[];
};

type Props = {
  redKey: string[];
  blueKey: string[];
  redCurrent?: CurrentRoundInfo | null;
  blueCurrent?: CurrentRoundInfo | null;
  currentRound?: number | null;
};

export default function DecryptoBoard({ 
  redKey, 
  blueKey, 
  redCurrent, 
  blueCurrent,
  currentRound 
}: Props) {
  return (
    <div className="decrypto-keys">
      {/* Red Team Board */}
      <div className="team-board red">
        <div className="team-board-header">
          <h4>Red Team</h4>
          {currentRound && redCurrent?.code?.length > 0 && (
            <span className="muted">Round {currentRound}</span>
          )}
        </div>
        <div className="team-board-content">
          <div className="key-cards">
            {redKey.map((word, idx) => (
              <div key={`red-${word}-${idx}`} className="key-card">
                <span className="key-num" style={{ background: 'var(--red-bg)', color: 'var(--red)' }}>{idx + 1}</span>
                {word}
              </div>
            ))}
          </div>
          {redCurrent && (redCurrent.code?.length > 0 || redCurrent.clues?.length > 0) && (
            <div className="current-info">
              {redCurrent.code?.length > 0 && (
                <div className="current-info-row">
                  <span className="current-info-label">Code:</span>
                  <span className="current-info-value">{redCurrent.code.join("-")}</span>
                </div>
              )}
              {redCurrent.clues?.length > 0 && (
                <div className="current-info-row">
                  <span className="current-info-label">Clues:</span>
                  <span className="current-info-value">{redCurrent.clues.join(" | ")}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Blue Team Board */}
      <div className="team-board blue">
        <div className="team-board-header">
          <h4>Blue Team</h4>
          {currentRound && blueCurrent?.code?.length > 0 && (
            <span className="muted">Round {currentRound}</span>
          )}
        </div>
        <div className="team-board-content">
          <div className="key-cards">
            {blueKey.map((word, idx) => (
              <div key={`blue-${word}-${idx}`} className="key-card">
                <span className="key-num" style={{ background: 'var(--blue-bg)', color: 'var(--blue)' }}>{idx + 1}</span>
                {word}
              </div>
            ))}
          </div>
          {blueCurrent && (blueCurrent.code?.length > 0 || blueCurrent.clues?.length > 0) && (
            <div className="current-info">
              {blueCurrent.code?.length > 0 && (
                <div className="current-info-row">
                  <span className="current-info-label">Code:</span>
                  <span className="current-info-value">{blueCurrent.code.join("-")}</span>
                </div>
              )}
              {blueCurrent.clues?.length > 0 && (
                <div className="current-info-row">
                  <span className="current-info-label">Clues:</span>
                  <span className="current-info-value">{blueCurrent.clues.join(" | ")}</span>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
