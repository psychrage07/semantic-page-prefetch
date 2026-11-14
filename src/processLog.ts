// The UI shows readable process messages. Full JSON dumps and the unchanged
// original process log remain available as downloaded evidence.
export function processMessages(log: string): string {
  return log.split(/\r?\n/).filter(line => {
    const text = line.trim();
    if (/^(?:Starting No prefetch|\[(?:none|stock)\])/.test(text)) return false;
    if (/^Completed .*strategy=(?:none|stock)(?::|\s|$)/.test(text)) return false;
    if (/^(?:[{}]|\[\s*$|\]|"[^\"]+"\s*:)/.test(text)) return false;
    if (/^(?:-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)[,\s]*$/.test(text)) return false;
    return true;
  }).join("\n").replace(/\n{3,}/g, "\n\n");
}
