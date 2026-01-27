import { HanabiCard, HanabiCardKnowledge } from "../types";

type Props = {
  playerOrder: string[];
  visibleHands: Record<string, HanabiCard[]>;
  myKnowledge: HanabiCardKnowledge[];
  currentPlayer: string;
  playedCards: Record<string, number>;
  discardPile: HanabiCard[];
  hintTokens: number;
  fuseTokens: number;
  score: number;
  deckRemaining: number;
  viewingAs?: string;
};

const COLORS = ["red", "yellow", "green", "blue", "white"];
const COLOR_CLASSES: Record<string, string> = {
  red: "hanabi-red",
  yellow: "hanabi-yellow",
  green: "hanabi-green",
  blue: "hanabi-blue",
  white: "hanabi-white",
};

function CardDisplay({ card, size = "normal" }: { card: HanabiCard; size?: "normal" | "small" }) {
  const colorClass = COLOR_CLASSES[card.color] || "";
  return (
    <div className={`hanabi-card ${colorClass} ${size}`}>
      <span className="card-number">{card.number}</span>
    </div>
  );
}

function KnowledgeCard({ knowledge, position }: { knowledge: HanabiCardKnowledge; position: number }) {
  const hasColor = knowledge.known_color !== null;
  const hasNumber = knowledge.known_number !== null;
  const colorClass = hasColor ? COLOR_CLASSES[knowledge.known_color!] : "hanabi-unknown";
  
  return (
    <div className={`hanabi-card knowledge ${colorClass}`} title={`Position ${position}`}>
      {hasNumber ? (
        <span className="card-number">{knowledge.known_number}</span>
      ) : (
        <span className="card-unknown">?</span>
      )}
      {!hasColor && !hasNumber && (
        <div className="knowledge-hints">
          <div className="possible-colors">
            {knowledge.possible_colors.map(c => (
              <span key={c} className={`color-dot ${COLOR_CLASSES[c]}`} />
            ))}
          </div>
          <div className="possible-numbers">
            {knowledge.possible_numbers.join("")}
          </div>
        </div>
      )}
    </div>
  );
}

function PlayerHand({ 
  playerId, 
  cards, 
  isCurrentPlayer,
  isViewer 
}: { 
  playerId: string; 
  cards: HanabiCard[]; 
  isCurrentPlayer: boolean;
  isViewer: boolean;
}) {
  return (
    <div className={`player-hand ${isCurrentPlayer ? "active" : ""} ${isViewer ? "viewer" : ""}`}>
      <div className="player-name">
        {playerId}
        {isCurrentPlayer && <span className="turn-indicator"> (current)</span>}
        {isViewer && <span className="viewer-indicator"> (you)</span>}
      </div>
      <div className="hand-cards">
        {cards.map((card, i) => (
          <CardDisplay key={i} card={card} />
        ))}
      </div>
    </div>
  );
}

function MyHand({ 
  knowledge, 
  isCurrentPlayer 
}: { 
  knowledge: HanabiCardKnowledge[]; 
  isCurrentPlayer: boolean;
}) {
  return (
    <div className={`player-hand my-hand ${isCurrentPlayer ? "active" : ""}`}>
      <div className="player-name">
        Your Hand
        {isCurrentPlayer && <span className="turn-indicator"> (your turn)</span>}
      </div>
      <div className="hand-cards">
        {knowledge.map((k, i) => (
          <KnowledgeCard key={i} knowledge={k} position={i} />
        ))}
      </div>
      <div className="hand-positions">
        {knowledge.map((_, i) => (
          <span key={i} className="position-label">{i}</span>
        ))}
      </div>
    </div>
  );
}

function PlayedStacks({ playedCards }: { playedCards: Record<string, number> }) {
  return (
    <div className="played-stacks">
      <h4>Fireworks</h4>
      <div className="stacks">
        {COLORS.map(color => {
          const num = playedCards[color] || 0;
          return (
            <div key={color} className={`stack ${COLOR_CLASSES[color]}`}>
              <div className="stack-label">{color[0].toUpperCase()}</div>
              <div className="stack-value">{num > 0 ? num : "—"}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DiscardPile({ cards }: { cards: HanabiCard[] }) {
  // Group by color and number for compact display
  const grouped: Record<string, number[]> = {};
  cards.forEach(card => {
    if (!grouped[card.color]) grouped[card.color] = [];
    grouped[card.color].push(card.number);
  });
  
  return (
    <div className="discard-pile">
      <h4>Discards ({cards.length})</h4>
      <div className="discard-groups">
        {Object.entries(grouped).map(([color, numbers]) => (
          <div key={color} className="discard-group">
            <span className={`color-label ${COLOR_CLASSES[color]}`}>{color[0].toUpperCase()}:</span>
            <span className="numbers">{numbers.sort().join(", ")}</span>
          </div>
        ))}
        {cards.length === 0 && <span className="empty">(empty)</span>}
      </div>
    </div>
  );
}

function Tokens({ hints, fuses, deck }: { hints: number; fuses: number; deck: number }) {
  return (
    <div className="tokens">
      <div className="token-group">
        <span className="token-label">Hints</span>
        <span className="token-value hint-tokens">
          {"💡".repeat(hints)}{"·".repeat(8 - hints)}
        </span>
        <span className="token-count">{hints}/8</span>
      </div>
      <div className="token-group">
        <span className="token-label">Fuses</span>
        <span className="token-value fuse-tokens">
          {"💥".repeat(fuses)}{"·".repeat(3 - fuses)}
        </span>
        <span className="token-count">{fuses}/3</span>
      </div>
      <div className="token-group">
        <span className="token-label">Deck</span>
        <span className="token-count deck-count">{deck}</span>
      </div>
    </div>
  );
}

export default function HanabiBoard({
  playerOrder,
  visibleHands,
  myKnowledge,
  currentPlayer,
  playedCards,
  discardPile,
  hintTokens,
  fuseTokens,
  score,
  deckRemaining,
  viewingAs,
}: Props) {
  return (
    <div className="hanabi-board">
      <div className="game-info">
        <div className="score-display">
          <span className="score-label">Score</span>
          <span className="score-value">{score}/25</span>
        </div>
        <Tokens hints={hintTokens} fuses={fuseTokens} deck={deckRemaining} />
      </div>
      
      <div className="hands-section">
        {/* Other players' visible hands */}
        <div className="other-hands">
          {playerOrder
            .filter(pid => pid !== viewingAs)
            .map(pid => (
              <PlayerHand
                key={pid}
                playerId={pid}
                cards={visibleHands[pid] || []}
                isCurrentPlayer={pid === currentPlayer}
                isViewer={false}
              />
            ))}
        </div>
        
        {/* Player's own hand (if viewing as a player) */}
        {viewingAs && myKnowledge.length > 0 && (
          <MyHand 
            knowledge={myKnowledge} 
            isCurrentPlayer={viewingAs === currentPlayer}
          />
        )}
      </div>
      
      <div className="table-section">
        <PlayedStacks playedCards={playedCards} />
        <DiscardPile cards={discardPile} />
      </div>
    </div>
  );
}
