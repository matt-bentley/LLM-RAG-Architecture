namespace Ai.Rag.Demo.Tests.Services;

public class TextTokenizerTests
{
    [Fact]
    public void Constructor_WithDefaultParameters_CreatesTokenizerWithEnglishStopwords()
    {
        // Arrange & Act
        var tokenizer = new TextTokenizer();

        // Assert
        var tokens = tokenizer.Tokenize("the quick brown fox");
        tokens.Should().NotContain("the");
        tokens.Should().Contain("quick");
        tokens.Should().Contain("brown");
        tokens.Should().Contain("fox");
    }

    [Fact]
    public void Constructor_WithCustomStopwords_UsesProvidedStopwords()
    {
        // Arrange
        var customStopwords = new[] { "custom", "stop" };
        var tokenizer = new TextTokenizer(customStopwords);

        // Act
        var tokens = tokenizer.Tokenize("custom word stop another");

        // Assert
        tokens.Should().NotContain("custom");
        tokens.Should().NotContain("stop");
        tokens.Should().Contain("word");
        tokens.Should().Contain("another");
    }

    [Fact]
    public void Constructor_WithPreserveArticleReferencesDisabled_DoesNotPreserveReferences()
    {
        // Arrange
        var tokenizer = new TextTokenizer(preserveArticleReferences: false);

        // Act
        var tokens = tokenizer.Tokenize("Article 4.3.2 states");

        // Assert
        tokens.Should().NotContain("article_4.3.2");
        tokens.Should().Contain("article");
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void Tokenize_WithNullOrWhitespace_ReturnsEmptyList(string input)
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize(input);

        // Assert
        tokens.Should().BeEmpty();
    }

    [Fact]
    public void Tokenize_WithSimpleText_ReturnsLowercaseTokens()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var tokens = tokenizer.Tokenize("Hello World TEST");

        // Assert
        tokens.Should().Contain("hello");
        tokens.Should().Contain("world");
        tokens.Should().Contain("test");
    }

    [Fact]
    public void Tokenize_RemovesPunctuation()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var tokens = tokenizer.Tokenize("Hello, World! How are you?");

        // Assert
        tokens.Should().Contain("hello");
        tokens.Should().Contain("world");
        tokens.Should().Contain("how");
        tokens.Should().Contain("are");
        tokens.Should().Contain("you");
        tokens.Should().NotContain(",");
        tokens.Should().NotContain("!");
        tokens.Should().NotContain("?");
    }

    [Fact]
    public void Tokenize_WithMinTokenLength_FiltersShortTokens()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None, minTokenLength: 4);

        // Act
        var tokens = tokenizer.Tokenize("a ab abc abcd abcde");

        // Assert
        tokens.Should().NotContain("a");
        tokens.Should().NotContain("ab");
        tokens.Should().NotContain("abc");
        tokens.Should().Contain("abcd");
        tokens.Should().Contain("abcde");
    }

    [Fact]
    public void Tokenize_WithStopwords_RemovesStopwords()
    {
        // Arrange
        var tokenizer = new TextTokenizer(); // Uses English stopwords

        // Act
        var tokens = tokenizer.Tokenize("the quick brown fox jumps over the lazy dog");

        // Assert
        tokens.Should().NotContain("the");
        tokens.Should().Contain("quick");
        tokens.Should().Contain("brown");
        tokens.Should().Contain("fox");
        tokens.Should().Contain("jumps");
        tokens.Should().Contain("lazy");
        tokens.Should().Contain("dog");
    }

    [Theory]
    [InlineData("Article 4.3.2", "article_4.3.2", "4.3.2")]
    [InlineData("Section 1.2", "section_1.2", "1.2")]
    [InlineData("Art. 5.1.3", "article_5.1.3", "5.1.3")]
    [InlineData("Sec. 2.4", "section_2.4", "2.4")]
    [InlineData("§ 3.2.1", "section_3.2.1", "3.2.1")]
    [InlineData("Paragraph 6", "paragraph_6", "6")]
    [InlineData("Para. 7.1", "paragraph_7.1", "7.1")]
    [InlineData("Rule 8.2.3", "rule_8.2.3", "8.2.3")]
    [InlineData("Regulation 9.1", "regulation_9.1", "9.1")]
    [InlineData("Reg. 10.2.4", "regulation_10.2.4", "10.2.4")]
    public void Tokenize_WithArticleReferences_PreservesAsCompoundTokens(string input, string expectedCompound, string expectedNumber)
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize(input);

        // Assert
        tokens.Should().Contain(expectedCompound);
        tokens.Should().Contain(expectedNumber);
    }

    [Fact]
    public void Tokenize_WithArticleReferenceWithParentheses_NormalizesCorrectly()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize("Article 4.3.2(a) applies");

        // Assert
        tokens.Should().Contain("article_4.3.2.a");
        tokens.Should().Contain("4.3.2.a");
    }

    [Fact]
    public void Tokenize_WithStandaloneNumberedReference_ExtractsCorrectly()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize("Observe 4.3.2 for details");

        // Assert
        tokens.Should().Contain("4.3.2");
        tokens.Should().Contain("observe");
        tokens.Should().Contain("details");
    }

    [Fact]
    public void Tokenize_WithMultipleArticleReferences_ExtractsAllReferences()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize("Article 4.3.2 and Section 1.2 refer to 5.6.7");

        // Assert
        tokens.Should().Contain("article_4.3.2");
        tokens.Should().Contain("4.3.2");
        tokens.Should().Contain("section_1.2");
        tokens.Should().Contain("1.2");
        tokens.Should().Contain("5.6.7");
    }

    [Fact]
    public void Tokenize_WithCaseInsensitiveArticleReferences_HandlesCorrectly()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var tokens = tokenizer.Tokenize("ARTICLE 1.2 and article 3.4");

        // Assert
        tokens.Should().Contain("article_1.2");
        tokens.Should().Contain("1.2");
        tokens.Should().Contain("article_3.4");
        tokens.Should().Contain("3.4");
    }

    [Fact]
    public void HashTermToIndex_WithSameTerm_ReturnsSameHash()
    {
        // Arrange
        var tokenizer = new TextTokenizer();
        var term = "test";

        // Act
        var hash1 = tokenizer.HashTermToIndex(term);
        var hash2 = tokenizer.HashTermToIndex(term);

        // Assert
        hash1.Should().Be(hash2);
    }

    [Fact]
    public void HashTermToIndex_WithDifferentTerms_ReturnsDifferentHashes()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var hash1 = tokenizer.HashTermToIndex("term1");
        var hash2 = tokenizer.HashTermToIndex("term2");

        // Assert
        hash1.Should().NotBe(hash2);
    }

    [Fact]
    public void HashTermToIndex_ReturnsUInt32Value()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var hash = tokenizer.HashTermToIndex("test");

        // Assert
        hash.Should().BeOfType(typeof(uint));
        hash.Should().BeGreaterThan(0u);
    }

    [Fact]
    public void GetTermFrequencies_WithEmptyText_ReturnsEmptyDictionary()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var (termFrequencies, tokenCount) = tokenizer.GetTermFrequencies("");

        // Assert
        termFrequencies.Should().BeEmpty();
        tokenCount.Should().Be(0);
    }

    [Fact]
    public void GetTermFrequencies_WithSimpleText_ReturnsCorrectFrequencies()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var (termFrequencies, tokenCount) = tokenizer.GetTermFrequencies("hello world hello");

        // Assert
        tokenCount.Should().Be(3);
        termFrequencies.Should().HaveCount(2);
        
        var helloHash = tokenizer.HashTermToIndex("hello");
        var worldHash = tokenizer.HashTermToIndex("world");
        
        termFrequencies[helloHash].Should().Be(2);
        termFrequencies[worldHash].Should().Be(1);
    }

    [Fact]
    public void GetTermFrequencies_WithRepeatedTerms_CountsCorrectly()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var (termFrequencies, tokenCount) = tokenizer.GetTermFrequencies("cat dog cat cat dog cat");

        // Assert
        tokenCount.Should().Be(6);
        
        var catHash = tokenizer.HashTermToIndex("cat");
        var dogHash = tokenizer.HashTermToIndex("dog");
        
        termFrequencies[catHash].Should().Be(4);
        termFrequencies[dogHash].Should().Be(2);
    }

    [Fact]
    public void GetTermFrequencies_WithArticleReferences_IncludesCompoundTokens()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var (termFrequencies, tokenCount) = tokenizer.GetTermFrequencies("Article 4.3.2 states Article 4.3.2");

        // Assert
        var compoundHash = tokenizer.HashTermToIndex("article_4.3.2");
        var numberHash = tokenizer.HashTermToIndex("4.3.2");
        
        termFrequencies[compoundHash].Should().Be(2);
        termFrequencies[numberHash].Should().Be(2);
    }

    [Fact]
    public void GetUniqueTermHashes_WithEmptyText_ReturnsEmptySet()
    {
        // Arrange
        var tokenizer = new TextTokenizer();

        // Act
        var hashes = tokenizer.GetUniqueTermHashes("");

        // Assert
        hashes.Should().BeEmpty();
    }

    [Fact]
    public void GetUniqueTermHashes_WithUniqueTerms_ReturnsAllHashes()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var hashes = tokenizer.GetUniqueTermHashes("hello world test");

        // Assert
        hashes.Should().HaveCount(3);
        hashes.Should().Contain(tokenizer.HashTermToIndex("hello"));
        hashes.Should().Contain(tokenizer.HashTermToIndex("world"));
        hashes.Should().Contain(tokenizer.HashTermToIndex("test"));
    }

    [Fact]
    public void GetUniqueTermHashes_WithRepeatedTerms_ReturnsUniqueHashesOnly()
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var hashes = tokenizer.GetUniqueTermHashes("hello world hello test world");

        // Assert
        hashes.Should().HaveCount(3);
        hashes.Should().Contain(tokenizer.HashTermToIndex("hello"));
        hashes.Should().Contain(tokenizer.HashTermToIndex("world"));
        hashes.Should().Contain(tokenizer.HashTermToIndex("test"));
    }

    [Fact]
    public void Tokenize_WithComplexLegalText_ExtractsAllReferences()
    {
        // Arrange
        var tokenizer = new TextTokenizer();
        var text = @"According to Article 4.3.2(a), Section 1.2, and § 5.6.7, 
                     the provisions in Regulation 10.2 apply. See also 8.9.10 and Para. 3.4.";

        // Act
        var tokens = tokenizer.Tokenize(text);

        // Assert
        tokens.Should().Contain("article_4.3.2.a");
        tokens.Should().Contain("4.3.2.a");
        tokens.Should().Contain("section_1.2");
        tokens.Should().Contain("1.2");
        tokens.Should().Contain("section_5.6.7");
        tokens.Should().Contain("5.6.7");
        tokens.Should().Contain("regulation_10.2");
        tokens.Should().Contain("10.2");
        tokens.Should().Contain("8.9.10");
        tokens.Should().Contain("paragraph_3.4");
        tokens.Should().Contain("3.4");
    }

    [Theory]
    [InlineData("Hello123World", "hello123world")]
    [InlineData("test_case", "test_case")]
    [InlineData("CamelCase", "camelcase")]
    public void Tokenize_WithMixedCaseAndNumbers_NormalizesCorrectly(string input, string expected)
    {
        // Arrange
        var tokenizer = new TextTokenizer(StopWordLists.None);

        // Act
        var tokens = tokenizer.Tokenize(input);

        // Assert
        tokens.Should().Contain(expected);
    }
}
