type Props = {
  words: string[];
  keyByWord: Record<string, string>;
  revealed: Record<string, string>;
};

function cardClass(cardType: string, revealed: boolean) {
  const base = "card";
  if (!cardType) return base;
  const typeClass = cardType.toLowerCase();
  return revealed ? `${base} revealed ${typeClass}` : `${base} ${typeClass}`;
}

export default function CodenamesBoard({ words, keyByWord, revealed }: Props) {
  return (
    <div className="board">
      {words.map((word) => {
        const keyType = keyByWord[word] || "NEUTRAL";
        const isRevealed = !!revealed[word];
        return (
          <div key={word} className={cardClass(keyType, isRevealed)}>
            <div className="card-word">{word}</div>
            <div className="card-type">{keyType}</div>
          </div>
        );
      })}
    </div>
  );
}
