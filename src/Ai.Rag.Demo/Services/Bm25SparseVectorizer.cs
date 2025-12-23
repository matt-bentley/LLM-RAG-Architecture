using System.Collections.Concurrent;
using System.Text.Json;

namespace Ai.Rag.Demo.Services;

/// <summary>
/// Computes BM25-weighted sparse vectors for text using hash-based token indices.
/// Maintains document frequency statistics for IDF calculation.
/// </summary>
public class Bm25SparseVectorizer
{
    // BM25 parameters
    private readonly double _k1;
    private readonly double _b;

    // Document frequency tracking for IDF calculation
    private readonly ConcurrentDictionary<uint, int> _documentFrequencies = new();
    private int _totalDocuments;
    private long _totalDocumentLength;
    private readonly object _dfLock = new();
    private readonly string _storagePath;

    // Tokenizer service
    private readonly TextTokenizer _tokenizer;

    /// <summary>
    /// Creates a new BM25 sparse vectorizer
    /// </summary>
    /// <param name="storagePath">Path to persist document frequency data. If null, frequencies are not persisted.</param>
    /// <param name="k1">BM25 k1 parameter controlling term frequency saturation (default 1.2)</param>
    /// <param name="b">BM25 b parameter controlling length normalization (default 0.75)</param>
    /// <param name="tokenizer">Optional tokenizer instance. If null, a new one is created.</param>
    public Bm25SparseVectorizer(
        string storagePath = null,
        double k1 = 1.2,
        double b = 0.75,
        TextTokenizer tokenizer = null)
    {
        _storagePath = storagePath;
        _k1 = k1;
        _b = b;
        _tokenizer = tokenizer ?? new TextTokenizer();

        LoadDocumentFrequencies();
    }

    /// <summary>
    /// Gets the total number of documents indexed
    /// </summary>
    public int TotalDocuments => _totalDocuments;

    /// <summary>
    /// Gets the number of unique term hashes tracked
    /// </summary>
    public int UniqueTermCount => _documentFrequencies.Count;

    /// <summary>
    /// Gets the average document length (in tokens)
    /// </summary>
    public double AverageDocumentLength => _totalDocuments > 0 ? (double)_totalDocumentLength / _totalDocuments : 1.0;

    /// <summary>
    /// Updates document frequency statistics for a new document.
    /// Call this once per document before computing sparse vectors.
    /// </summary>
    /// <param name="text">The document text</param>
    public void AddDocument(string text)
    {
        var tokens = _tokenizer.Tokenize(text);
        if (tokens.Count == 0) return;

        // Get unique term hashes in this document
        var uniqueHashes = tokens.Select(t => _tokenizer.HashTermToIndex(t)).Distinct().ToList();

        lock (_dfLock)
        {
            _totalDocuments++;
            _totalDocumentLength += tokens.Count;

            foreach (var hash in uniqueHashes)
            {
                _documentFrequencies.AddOrUpdate(hash, 1, (_, count) => count + 1);
            }
        }
    }

    /// <summary>
    /// Updates document frequencies for multiple documents and persists the changes
    /// </summary>
    /// <param name="texts">The document texts</param>
    public void AddDocuments(IEnumerable<string> texts)
    {
        foreach (var text in texts)
        {
            AddDocument(text);
        }
        SaveDocumentFrequencies();
    }

    /// <summary>
    /// Computes a BM25-weighted sparse vector for the given text
    /// </summary>
    /// <param name="text">The text to vectorize</param>
    /// <returns>Tuple of (indices, values) representing the sparse vector</returns>
    public (uint[] Indices, float[] Values) ComputeSparseVector(string text)
    {
        var (termFrequencies, tokenCount) = _tokenizer.GetTermFrequencies(text);
        if (tokenCount == 0)
        {
            return ([], []);
        }

        var avgDocLength = AverageDocumentLength;

        // Calculate BM25 term weights with IDF
        var sparseEntries = new Dictionary<uint, float>();
        var totalDocs = Math.Max(1, _totalDocuments);

        foreach (var (index, tf) in termFrequencies)
        {
            // Get document frequency for this term
            var df = _documentFrequencies.GetValueOrDefault(index, 0);

            // IDF calculation: log((N - df + 0.5) / (df + 0.5) + 1)
            // This is the BM25 IDF formula (with +1 to avoid negative values for very common terms)
            var idf = Math.Log((totalDocs - df + 0.5) / (df + 0.5) + 1);

            // BM25 term frequency saturation formula
            // TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (docLength / avgDocLength)))
            var tfNormalized = (tf * (_k1 + 1)) / (tf + _k1 * (1 - _b + _b * (tokenCount / avgDocLength)));

            // Full BM25 weight = IDF * TF_normalized
            var bm25Weight = idf * tfNormalized;

            sparseEntries[index] = (float)bm25Weight;
        }

        var sortedEntries = sparseEntries.OrderBy(e => e.Key).ToList();
        return (sortedEntries.Select(e => e.Key).ToArray(), sortedEntries.Select(e => e.Value).ToArray());
    }

    /// <summary>
    /// Clears all document frequency statistics
    /// </summary>
    public void Clear()
    {
        lock (_dfLock)
        {
            _documentFrequencies.Clear();
            _totalDocuments = 0;
            _totalDocumentLength = 0;
        }
        SaveDocumentFrequencies();
    }

    /// <summary>
    /// Persists document frequencies to disk
    /// </summary>
    public void SaveDocumentFrequencies()
    {
        if (string.IsNullOrEmpty(_storagePath)) return;

        try
        {
            var data = new DocumentFrequencyData
            {
                TotalDocuments = _totalDocuments,
                TotalDocumentLength = _totalDocumentLength,
                Frequencies = _documentFrequencies.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
            };
            var json = JsonSerializer.Serialize(data);
            File.WriteAllText(_storagePath, json);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to save document frequencies: {ex.Message}");
        }
    }

    /// <summary>
    /// Loads document frequencies from disk
    /// </summary>
    private void LoadDocumentFrequencies()
    {
        if (string.IsNullOrEmpty(_storagePath)) return;

        try
        {
            if (File.Exists(_storagePath))
            {
                var json = File.ReadAllText(_storagePath);
                var data = JsonSerializer.Deserialize<DocumentFrequencyData>(json);
                if (data != null)
                {
                    _totalDocuments = data.TotalDocuments;
                    _totalDocumentLength = data.TotalDocumentLength;
                    foreach (var kvp in data.Frequencies)
                    {
                        _documentFrequencies[kvp.Key] = kvp.Value;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Warning: Failed to load document frequencies: {ex.Message}");
        }
    }

    /// <summary>
    /// Data structure for persisting document frequencies
    /// </summary>
    private class DocumentFrequencyData
    {
        public int TotalDocuments { get; set; }
        public long TotalDocumentLength { get; set; }
        public Dictionary<uint, int> Frequencies { get; set; } = new();
    }
}
