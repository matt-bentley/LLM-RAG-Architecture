using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Service for tokenizing text and hashing tokens to sparse vector indices.
/// Used for BM25 sparse vector computation.
/// </summary>
public class TextTokenizer
{
    private readonly HashSet<string> _stopWords;
    private readonly int _minTokenLength;
    private readonly bool _preserveArticleReferences;

    // Regex for tokenization - compiled for performance
    private static readonly Regex TokenizerRegex =
        new(@"[^\w]+", RegexOptions.Compiled);

    // Pattern to match article/section references like "Article 4.3.2", "Section 1.2", "§ 4.3.2"
    // Captures the type (Article/Section/§/Art./Sec.) and the number with dots
    private static readonly Regex ArticleReferenceRegex =
        new(@"(?<type>Article|Section|Art\.?|Sec\.?|§|Paragraph|Para\.?|Rule|Regulation|Reg\.?)\s*(?<number>\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))?)",
            RegexOptions.Compiled | RegexOptions.IgnoreCase);

    // Pattern for standalone numbered references like "4.3.2" or "1.2.3(a)"
    private static readonly Regex NumberedReferenceRegex =
        new(@"\b(?<number>\d+(?:\.\d+)+(?:\([a-zA-Z0-9]+\))?)\b",
            RegexOptions.Compiled);

    /// <summary>
    /// Creates a new text tokenizer with default English stopwords
    /// </summary>
    /// <param name="preserveArticleReferences">If true, preserves article/section references as single tokens (default true)</param>
    public TextTokenizer(bool preserveArticleReferences = true) 
        : this(StopWordLists.English, preserveArticleReferences: preserveArticleReferences)
    {
    }

    /// <summary>
    /// Creates a new text tokenizer with custom stopwords
    /// </summary>
    /// <param name="stopWords">Set of stopwords to filter out. Pass empty set to disable stopword filtering.</param>
    /// <param name="minTokenLength">Minimum token length to include (default 2)</param>
    /// <param name="preserveArticleReferences">If true, preserves article/section references as single tokens (default true)</param>
    public TextTokenizer(
        IEnumerable<string> stopWords, 
        int minTokenLength = 2,
        bool preserveArticleReferences = true)
    {
        _stopWords = new HashSet<string>(stopWords, StringComparer.OrdinalIgnoreCase);
        _minTokenLength = minTokenLength;
        _preserveArticleReferences = preserveArticleReferences;
    }

    /// <summary>
    /// Tokenizes text into lowercase terms, removing punctuation and stopwords.
    /// Preserves article/section references as compound tokens if enabled.
    /// </summary>
    /// <param name="text">The text to tokenize</param>
    /// <returns>List of tokens</returns>
    public List<string> Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return [];
        }

        var tokens = new List<string>();
        var processedText = text;

        if (_preserveArticleReferences)
        {
            // Extract and preserve article references before standard tokenization
            var (extractedTokens, remainingText) = ExtractArticleReferences(processedText);
            tokens.AddRange(extractedTokens);
            processedText = remainingText;
        }

        // Standard tokenization for remaining text
        var standardTokens = TokenizerRegex.Split(processedText.ToLowerInvariant())
            .Where(t => !string.IsNullOrWhiteSpace(t) && t.Length >= _minTokenLength)
            .Where(t => !_stopWords.Contains(t));

        tokens.AddRange(standardTokens);

        return tokens;
    }

    /// <summary>
    /// Extracts article/section references from text and returns them as normalized tokens
    /// </summary>
    private (List<string> Tokens, string RemainingText) ExtractArticleReferences(string text)
    {
        var tokens = new List<string>();
        var remainingText = text;

        // First, extract full article references (e.g., "Article 4.3.2")
        var articleMatches = ArticleReferenceRegex.Matches(remainingText);
        foreach (Match match in articleMatches)
        {
            var type = NormalizeReferenceType(match.Groups["type"].Value);
            var number = NormalizeReferenceNumber(match.Groups["number"].Value);
            
            // Create compound token: "article_4.3.2"
            tokens.Add($"{type}_{number}");
            
            // Also add the number alone for flexible matching: "4.3.2"
            tokens.Add(number);
        }

        // Remove matched references from text to avoid double-processing
        remainingText = ArticleReferenceRegex.Replace(remainingText, " ");

        // Then, extract standalone numbered references (e.g., "4.3.2" not preceded by Article/Section)
        var numberMatches = NumberedReferenceRegex.Matches(remainingText);
        foreach (Match match in numberMatches)
        {
            var number = NormalizeReferenceNumber(match.Groups["number"].Value);
            tokens.Add(number);
        }

        // Remove standalone numbered references
        remainingText = NumberedReferenceRegex.Replace(remainingText, " ");

        return (tokens, remainingText);
    }

    /// <summary>
    /// Normalizes reference types to standard forms
    /// </summary>
    private static string NormalizeReferenceType(string type)
    {
        return type.ToLowerInvariant().TrimEnd('.') switch
        {
            "article" or "art" => "article",
            "section" or "sec" => "section",
            "paragraph" or "para" => "paragraph",
            "rule" => "rule",
            "regulation" or "reg" => "regulation",
            "§" => "section",
            _ => type.ToLowerInvariant().TrimEnd('.')
        };
    }

    /// <summary>
    /// Normalizes reference numbers (removes parentheses formatting variations)
    /// </summary>
    private static string NormalizeReferenceNumber(string number)
    {
        // Normalize parenthetical parts: 4.3.2(a) -> 4.3.2.a
        return Regex.Replace(number.ToLowerInvariant(), @"\(([a-zA-Z0-9]+)\)", ".$1");
    }

    /// <summary>
    /// Hashes a term to a 32-bit unsigned integer index for sparse vector representation
    /// </summary>
    /// <param name="term">The term to hash</param>
    /// <returns>A 32-bit unsigned integer index</returns>
    public uint HashTermToIndex(string term)
    {
        var bytes = Encoding.UTF8.GetBytes(term);
        var hash = MD5.HashData(bytes);
        return BitConverter.ToUInt32(hash, 0);
    }

    /// <summary>
    /// Tokenizes text and returns term frequencies with their hashed indices
    /// </summary>
    /// <param name="text">The text to process</param>
    /// <returns>Dictionary mapping hashed indices to term frequencies, plus token count</returns>
    public (Dictionary<uint, int> TermFrequencies, int TokenCount) GetTermFrequencies(string text)
    {
        var tokens = Tokenize(text);
        if (tokens.Count == 0)
        {
            return (new Dictionary<uint, int>(), 0);
        }

        var termFrequencies = new Dictionary<uint, int>();

        foreach (var token in tokens)
        {
            var index = HashTermToIndex(token);
            if (termFrequencies.TryGetValue(index, out var count))
            {
                termFrequencies[index] = count + 1;
            }
            else
            {
                termFrequencies[index] = 1;
            }
        }

        return (termFrequencies, tokens.Count);
    }

    /// <summary>
    /// Gets unique term hashes for a document (for document frequency tracking)
    /// </summary>
    /// <param name="text">The text to process</param>
    /// <returns>Set of unique term hashes</returns>
    public HashSet<uint> GetUniqueTermHashes(string text)
    {
        var tokens = Tokenize(text);
        return tokens.Select(HashTermToIndex).ToHashSet();
    }
}

/// <summary>
/// Pre-defined stopword lists for common languages
/// </summary>
public static class StopWordLists
{
    /// <summary>
    /// Creates a tokenizer with no stopword filtering
    /// </summary>
    public static IEnumerable<string> None => [];

    /// <summary>
    /// Common English stopwords
    /// </summary>
    public static IEnumerable<string> English =>
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "this", "but", "they",
        "have", "had", "what", "when", "where", "who", "which", "why", "how",
        "all", "each", "both", "few", "more", "most", "some",
        "such", "no", "nor", "not", "only", "own", "so", "than", "too",
        "very", "can", "just", "should", "now", "or", "if", "then", "else",
        "there", "here", "about", "into",
        "she", "her", "him", "his", "we", "our", "you", "your", "me", "my",
        "been", "being", "do", "does", "did", "doing", "would", "could", "might",
        "must", "shall", "may", "am", "also", "any", "because", "become",
        "becomes", "becoming", "come", "comes", "came", "get", "gets", "got",
        "go", "goes", "went", "going",
        "see", "seen", "saw", "took", "taking", "know",
        "knows", "knew", "known", "think", "thinks", "thought", "want",
        "wants", "wanted", "use", "uses", "used", "using", "find", "finds",
        "found", "give", "gives", "gave", "given", "tell", "tells", "told",
        "say", "says", "said", "let", "put", "seem", "seems", "seemed"
    ];

    /// <summary>
    /// Extended English stopwords including more common words
    /// </summary>
    public static IEnumerable<string> EnglishExtended =>
        English.Concat([
            "actually", "already", "although", "always", "another", "anything",
            "anywhere", "around", "away", "back", "become", "behind", "best",
            "better", "big", "bring", "call", "change", "come", "consider",
            "day", "different", "down", "end", "enough", "even", "example",
            "fact", "far", "feel", "first", "follow", "general", "good", "great",
            "group", "hand", "help", "high", "home", "however", "important",
            "include", "instead", "keep", "kind", "large", "last", "later",
            "least", "less", "life", "like", "likely", "little", "long", "look",
            "lot", "many", "matter", "mean", "much", "need", "never", "new",
            "next", "nothing", "number", "often", "old", "one", "order", "own",
            "part", "particular", "people", "perhaps", "place", "point", "possible",
            "present", "problem", "public", "put", "quite", "rather", "really",
            "right", "same", "set", "several", "show", "since", "small", "something",
            "still", "sure", "system", "thing", "three", "time", "today", "together",
            "toward", "try", "turn", "two", "understand", "upon", "us", "usually",
            "way", "well", "while", "within", "without", "work", "world", "year", "yet"
        ]);

    /// <summary>
    /// German stopwords
    /// </summary>
    public static IEnumerable<string> German =>
    [
        "aber", "alle", "allem", "allen", "aller", "alles", "als", "also", "am",
        "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes",
        "auch", "auf", "aus", "bei", "bin", "bis", "bist", "da", "damit",
        "dann", "das", "dass", "daß", "dein", "deine", "dem", "den", "denn",
        "der", "des", "dessen", "die", "dies", "diese", "dieselbe", "dieselben",
        "diesem", "diesen", "dieser", "dieses", "doch", "dort", "du", "durch",
        "ein", "eine", "einem", "einen", "einer", "eines", "einig", "einige",
        "einigem", "einigen", "einiger", "einiges", "einmal", "er", "es", "etwas",
        "euch", "euer", "eure", "für", "gegen", "gewesen", "hab", "habe", "haben",
        "hat", "hatte", "hatten", "hier", "hin", "hinter", "ich", "ihm", "ihn",
        "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "im", "in", "indem",
        "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes", "jene", "jenem",
        "jenen", "jener", "jenes", "jetzt", "kann", "kein", "keine", "keinem",
        "keinen", "keiner", "können", "könnte", "machen", "man", "manche",
        "manchem", "manchen", "mancher", "manches", "mein", "meine", "meinem",
        "meinen", "meiner", "mir", "mit", "muss", "musste", "nach", "nicht",
        "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein",
        "seine", "seinem", "seinen", "seiner", "selbst", "sich", "sie", "sind",
        "so", "solche", "solchem", "solchen", "solcher", "soll", "sollte",
        "sondern", "sonst", "über", "um", "und", "uns", "unser", "unsere",
        "unter", "viel", "vom", "von", "vor", "während", "war", "waren", "warst",
        "was", "weg", "weil", "weiter", "welche", "welchem", "welchen", "welcher",
        "welches", "wenn", "werde", "werden", "wie", "wieder", "will", "wir",
        "wird", "wirst", "wo", "wollen", "wollte", "würde", "würden", "zu",
        "zum", "zur", "zwar", "zwischen"
    ];

    /// <summary>
    /// French stopwords
    /// </summary>
    public static IEnumerable<string> French =>
    [
        "ai", "aie", "aient", "aies", "ait", "as", "au", "aura", "aurai",
        "auraient", "aurais", "aurait", "auras", "aurez", "auriez", "aurions",
        "aurons", "auront", "aux", "avaient", "avais", "avait", "avec", "avez",
        "aviez", "avions", "avons", "ayant", "ayez", "ayons", "ce", "ceci",
        "cela", "celà", "ces", "cet", "cette", "dans", "de", "des", "du", "elle",
        "elles", "en", "es", "est", "et", "eu", "eue", "eues", "eurent", "eus",
        "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux",
        "eûmes", "eût", "eûtes", "furent", "fus", "fusse", "fussent", "fusses",
        "fussiez", "fussions", "fut", "fûmes", "fût", "fûtes", "ici", "il",
        "ils", "je", "la", "le", "les", "leur", "leurs", "lui", "ma", "mais",
        "me", "mes", "moi", "mon", "même", "ne", "nos", "notre", "nous", "on",
        "ont", "ou", "où", "par", "pas", "pour", "qu", "que", "quel", "quelle",
        "quelles", "quels", "qui", "sa", "sans", "se", "sera", "serai",
        "seraient", "serais", "serait", "seras", "serez", "seriez", "serions",
        "serons", "seront", "ses", "soi", "soient", "sois", "soit", "sommes",
        "son", "sont", "soyez", "soyons", "suis", "sur", "ta", "te", "tes",
        "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "étaient",
        "étais", "était", "étant", "étiez", "étions", "été", "étée", "étées",
        "étés", "êtes"
    ];

    /// <summary>
    /// Spanish stopwords
    /// </summary>
    public static IEnumerable<string> Spanish =>
    [
        "a", "al", "algo", "alguno", "alguna", "algunos", "algunas", "ante",
        "antes", "como", "con", "contra", "cual", "cuales", "cuando", "de",
        "del", "desde", "donde", "e", "el", "ella", "ellas", "ellos", "en",
        "entre", "era", "esa", "esas", "ese", "eso", "esos", "esta", "estas",
        "este", "esto", "estos", "fue", "fueron", "ha", "han", "hasta", "hay",
        "la", "las", "le", "les", "lo", "los", "mas", "más", "me", "mi", "mis",
        "muy", "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestras",
        "nuestro", "nuestros", "o", "os", "otra", "otras", "otro", "otros",
        "para", "pero", "poco", "por", "porque", "que", "quien", "quienes",
        "se", "sea", "sean", "según", "ser", "si", "sido", "sin", "sino", "so",
        "sobre", "son", "soy", "su", "sus", "tal", "también", "tan", "tanto",
        "te", "tengo", "ti", "tiene", "tienen", "toda", "todas", "todavía",
        "todo", "todos", "tu", "tus", "un", "una", "unas", "uno", "unos",
        "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros",
        "y", "ya", "yo"
    ];

    /// <summary>
    /// Combines multiple stopword lists
    /// </summary>
    public static IEnumerable<string> Combine(params IEnumerable<string>[] lists) =>
        lists.SelectMany(l => l).Distinct();

    /// <summary>
    /// Creates a custom stopword list by extending an existing one
    /// </summary>
    public static IEnumerable<string> Extend(IEnumerable<string> baseList, params string[] additionalWords) =>
        baseList.Concat(additionalWords).Distinct();

    /// <summary>
    /// Creates a custom stopword list by removing words from an existing one
    /// </summary>
    public static IEnumerable<string> Exclude(IEnumerable<string> baseList, params string[] wordsToKeep)
    {
        var exclusions = new HashSet<string>(wordsToKeep, StringComparer.OrdinalIgnoreCase);
        return baseList.Where(w => !exclusions.Contains(w));
    }

    /// <summary>
    /// Loads stopwords from a file (one word per line)
    /// </summary>
    public static IEnumerable<string> FromFile(string filePath)
    {
        if (!File.Exists(filePath))
        {
            return [];
        }

        return File.ReadAllLines(filePath)
            .Select(line => line.Trim())
            .Where(line => !string.IsNullOrWhiteSpace(line) && !line.StartsWith('#'));
    }
}
