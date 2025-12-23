namespace Ai.Rag.Demo.Tests.Services;

public class StopWordListsTests
{
    [Fact]
    public void None_ReturnsEmptyList()
    {
        // Act
        var stopWords = StopWordLists.None.ToList();

        // Assert
        stopWords.Should().BeEmpty();
    }

    [Fact]
    public void English_ContainsCommonStopwords()
    {
        // Act
        var stopWords = StopWordLists.English.ToList();

        // Assert
        stopWords.Should().NotBeEmpty();
        stopWords.Should().Contain("the");
        stopWords.Should().Contain("and");
        stopWords.Should().Contain("is");
        stopWords.Should().Contain("of");
        stopWords.Should().Contain("to");
    }

    [Fact]
    public void EnglishExtended_ContainsMoreWordsThanEnglish()
    {
        // Act
        var english = StopWordLists.English.ToList();
        var englishExtended = StopWordLists.EnglishExtended.ToList();

        // Assert
        englishExtended.Should().HaveCountGreaterThan(english.Count);
        englishExtended.Should().Contain(english);
    }

    [Fact]
    public void EnglishExtended_ContainsAdditionalCommonWords()
    {
        // Act
        var stopWords = StopWordLists.EnglishExtended.ToList();

        // Assert
        stopWords.Should().Contain("actually");
        stopWords.Should().Contain("already");
        stopWords.Should().Contain("important");
        stopWords.Should().Contain("example");
    }

    [Fact]
    public void German_ContainsCommonGermanStopwords()
    {
        // Act
        var stopWords = StopWordLists.German.ToList();

        // Assert
        stopWords.Should().NotBeEmpty();
        stopWords.Should().Contain("der");
        stopWords.Should().Contain("die");
        stopWords.Should().Contain("das");
        stopWords.Should().Contain("und");
        stopWords.Should().Contain("ist");
    }

    [Fact]
    public void French_ContainsCommonFrenchStopwords()
    {
        // Act
        var stopWords = StopWordLists.French.ToList();

        // Assert
        stopWords.Should().NotBeEmpty();
        stopWords.Should().Contain("le");
        stopWords.Should().Contain("la");
        stopWords.Should().Contain("et");
        stopWords.Should().Contain("est");
        stopWords.Should().Contain("de");
    }

    [Fact]
    public void Spanish_ContainsCommonSpanishStopwords()
    {
        // Act
        var stopWords = StopWordLists.Spanish.ToList();

        // Assert
        stopWords.Should().NotBeEmpty();
        stopWords.Should().Contain("el");
        stopWords.Should().Contain("la");
        stopWords.Should().Contain("de");
        stopWords.Should().Contain("y");
        stopWords.Should().Contain("en");
    }

    [Fact]
    public void Combine_WithMultipleLists_MergesAllWords()
    {
        // Act
        var combined = StopWordLists.Combine(
            new[] { "word1", "word2" },
            new[] { "word3", "word4" }
        ).ToList();

        // Assert
        combined.Should().HaveCount(4);
        combined.Should().Contain("word1");
        combined.Should().Contain("word2");
        combined.Should().Contain("word3");
        combined.Should().Contain("word4");
    }

    [Fact]
    public void Combine_WithDuplicateWords_ReturnsDistinctWords()
    {
        // Act
        var combined = StopWordLists.Combine(
            new[] { "word1", "word2", "duplicate" },
            new[] { "word3", "duplicate" }
        ).ToList();

        // Assert
        combined.Should().HaveCount(4);
        combined.Count(w => w == "duplicate").Should().Be(1);
    }

    [Fact]
    public void Combine_WithMultipleLanguages_CombinesAllStopwords()
    {
        // Act
        var combined = StopWordLists.Combine(
            StopWordLists.English.Take(5),
            StopWordLists.German.Take(5),
            StopWordLists.French.Take(5)
        ).ToList();

        // Assert
        combined.Should().HaveCountGreaterOrEqualTo(15); // At most 15 if no duplicates
    }

    [Fact]
    public void Extend_AddsWordsToBaseList()
    {
        // Arrange
        var baseList = new[] { "word1", "word2" };

        // Act
        var extended = StopWordLists.Extend(baseList, "word3", "word4").ToList();

        // Assert
        extended.Should().HaveCount(4);
        extended.Should().Contain("word1");
        extended.Should().Contain("word2");
        extended.Should().Contain("word3");
        extended.Should().Contain("word4");
    }

    [Fact]
    public void Extend_WithDuplicates_ReturnsDistinctWords()
    {
        // Arrange
        var baseList = new[] { "word1", "word2" };

        // Act
        var extended = StopWordLists.Extend(baseList, "word2", "word3").ToList();

        // Assert
        extended.Should().HaveCount(3);
        extended.Count(w => w == "word2").Should().Be(1);
    }

    [Fact]
    public void Extend_WithEnglishStopwords_AddsCustomWords()
    {
        // Act
        var extended = StopWordLists.Extend(
            StopWordLists.English,
            "custom1",
            "custom2"
        ).ToList();

        // Assert
        extended.Should().Contain("the");
        extended.Should().Contain("custom1");
        extended.Should().Contain("custom2");
    }

    [Fact]
    public void Exclude_RemovesSpecifiedWords()
    {
        // Arrange
        var baseList = new[] { "word1", "word2", "word3", "word4" };

        // Act
        var filtered = StopWordLists.Exclude(baseList, "word2", "word4").ToList();

        // Assert
        filtered.Should().HaveCount(2);
        filtered.Should().Contain("word1");
        filtered.Should().Contain("word3");
        filtered.Should().NotContain("word2");
        filtered.Should().NotContain("word4");
    }

    [Fact]
    public void Exclude_IsCaseInsensitive()
    {
        // Arrange
        var baseList = new[] { "Word1", "WORD2", "word3" };

        // Act
        var filtered = StopWordLists.Exclude(baseList, "word1", "word2").ToList();

        // Assert
        filtered.Should().HaveCount(1);
        filtered.Should().Contain("word3");
    }

    [Fact]
    public void Exclude_WithEnglishStopwords_RemovesSpecifiedWords()
    {
        // Arrange
        var english = StopWordLists.English.ToList();
        var originalCount = english.Count;

        // Act
        var filtered = StopWordLists.Exclude(StopWordLists.English, "the", "and", "is").ToList();

        // Assert
        filtered.Should().HaveCount(originalCount - 3);
        filtered.Should().NotContain("the");
        filtered.Should().NotContain("and");
        filtered.Should().NotContain("is");
    }

    [Fact]
    public void FromFile_WithNonExistentFile_ReturnsEmptyList()
    {
        // Act
        var stopWords = StopWordLists.FromFile("nonexistent.txt").ToList();

        // Assert
        stopWords.Should().BeEmpty();
    }

    [Fact]
    public void FromFile_WithValidFile_ReadsWordsCorrectly()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "word1\nword2\nword3\n");

            // Act
            var stopWords = StopWordLists.FromFile(tempFile).ToList();

            // Assert
            stopWords.Should().HaveCount(3);
            stopWords.Should().Contain("word1");
            stopWords.Should().Contain("word2");
            stopWords.Should().Contain("word3");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromFile_IgnoresEmptyLines()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "word1\n\nword2\n  \nword3\n");

            // Act
            var stopWords = StopWordLists.FromFile(tempFile).ToList();

            // Assert
            stopWords.Should().HaveCount(3);
            stopWords.Should().Contain("word1");
            stopWords.Should().Contain("word2");
            stopWords.Should().Contain("word3");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromFile_IgnoresCommentLines()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "word1\n# This is a comment\nword2\n#Another comment\nword3\n");

            // Act
            var stopWords = StopWordLists.FromFile(tempFile).ToList();

            // Assert
            stopWords.Should().HaveCount(3);
            stopWords.Should().Contain("word1");
            stopWords.Should().Contain("word2");
            stopWords.Should().Contain("word3");
            stopWords.Should().NotContain(w => w.StartsWith('#'));
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void FromFile_TrimsWhitespace()
    {
        // Arrange
        var tempFile = Path.GetTempFileName();
        try
        {
            File.WriteAllText(tempFile, "  word1  \n\t word2 \t\n   word3   \n");

            // Act
            var stopWords = StopWordLists.FromFile(tempFile).ToList();

            // Assert
            stopWords.Should().HaveCount(3);
            stopWords.Should().Contain("word1");
            stopWords.Should().Contain("word2");
            stopWords.Should().Contain("word3");
        }
        finally
        {
            if (File.Exists(tempFile))
                File.Delete(tempFile);
        }
    }

    [Fact]
    public void AllLanguageLists_AreNotEmpty()
    {
        // Act & Assert
        StopWordLists.English.Should().NotBeEmpty();
        StopWordLists.EnglishExtended.Should().NotBeEmpty();
        StopWordLists.German.Should().NotBeEmpty();
        StopWordLists.French.Should().NotBeEmpty();
        StopWordLists.Spanish.Should().NotBeEmpty();
    }
}
