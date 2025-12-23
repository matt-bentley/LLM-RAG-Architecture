using System.Text;
using UglyToad.PdfPig.Content;

namespace Ai.Rag.Demo.ExtractionStrategies;

/// <summary>
/// Shared utilities for PDF text extraction with line break detection.
/// </summary>
public static class PdfTextUtilities
{
    /// <summary>
    /// Extracts word positions from a PDF page for line break detection.
    /// </summary>
    /// <param name="words">The words from a PDF page.</param>
    /// <returns>A list of word positions with text and vertical coordinates.</returns>
    public static List<WordPosition> GetWordPositions(IEnumerable<Word> words)
    {
        return words.Select(w => new WordPosition
        {
            Text = w.Text,
            Top = w.BoundingBox.Top,
            Bottom = w.BoundingBox.Bottom
        }).ToList();
    }

    /// <summary>
    /// Builds text from word positions, inserting line breaks where appropriate.
    /// Uses heuristics based on vertical position changes to detect line breaks.
    /// </summary>
    /// <param name="words">The word positions to build text from.</param>
    /// <returns>The text with appropriate line breaks inserted.</returns>
    public static string BuildTextWithLineBreaks(List<WordPosition> words)
    {
        if (words == null || words.Count == 0)
        {
            return string.Empty;
        }

        var result = new StringBuilder();

        for (int i = 0; i < words.Count; i++)
        {
            var currentWord = words[i];

            if (i > 0)
            {
                var previousWord = words[i - 1];

                if (ShouldInsertLineBreak(previousWord, currentWord))
                {
                    result.AppendLine();
                }
                else
                {
                    result.Append(' ');
                }
            }

            result.Append(currentWord.Text);
        }

        return result.ToString();
    }

    /// <summary>
    /// Determines whether a line break should be inserted between two words.
    /// Uses heuristics to detect actual paragraph/section breaks, not wrapped text.
    /// </summary>
    /// <param name="previousWord">The previous word position.</param>
    /// <param name="currentWord">The current word position.</param>
    /// <returns>True if a line break should be inserted, false otherwise.</returns>
    public static bool ShouldInsertLineBreak(WordPosition previousWord, WordPosition currentWord)
    {
        // In PDF coordinates, Y typically increases upward, so:
        // - A new line below would have a smaller Top value
        // - Words on the same line should have similar Top/Bottom values

        // Calculate the line height of the previous word as a reference
        var previousLineHeight = previousWord.Top - previousWord.Bottom;
        var currentLineHeight = currentWord.Top - currentWord.Bottom;

        // Use the average line height as a baseline for comparison
        var avgLineHeight = (previousLineHeight + currentLineHeight) / 2;

        // If line height is too small, use a default threshold
        if (avgLineHeight < 1)
        {
            avgLineHeight = 10;
        }

        // Check if words are on the same line (similar vertical position)
        var baselineDifference = Math.Abs(previousWord.Bottom - currentWord.Bottom);
        var topDifference = Math.Abs(previousWord.Top - currentWord.Top);

        // If both top and bottom are similar, words are on the same line
        if (baselineDifference < avgLineHeight * 0.3 && topDifference < avgLineHeight * 0.3)
        {
            return false;
        }

        // Calculate vertical gap - positive means current word is below previous
        var verticalGap = previousWord.Bottom - currentWord.Top;

        // If there's a large gap (more than 1.5x line height), it's likely a paragraph break
        if (verticalGap > avgLineHeight * 1.5)
        {
            return true;
        }

        // For normal line wrapping (gap roughly equal to line height), 
        // only add line break if previous word ends with paragraph-ending punctuation
        if (verticalGap > avgLineHeight * 0.3)
        {
            // Check if previous word ends with punctuation that typically ends a paragraph
            var prevText = previousWord.Text.TrimEnd();
            if (string.IsNullOrEmpty(prevText))
            {
                return false;
            }

            var lastChar = prevText[^1];

            // Only add line break after sentence-ending punctuation or special cases
            // This includes: period, question mark, exclamation, colon (for headings/lists)
            if (lastChar is '.' or '?' or '!' or ':' or ';')
            {
                return true;
            }

            // Also check for list items or bullet points (numbers followed by period/paren)
            if (char.IsDigit(prevText[0]) && (prevText.EndsWith('.') || prevText.EndsWith(')')))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Represents a word's text and vertical position for line break detection.
    /// </summary>
    public sealed class WordPosition
    {
        /// <summary>
        /// The text content of the word.
        /// </summary>
        public string Text { get; init; } = string.Empty;

        /// <summary>
        /// The top Y coordinate of the word's bounding box.
        /// </summary>
        public double Top { get; init; }

        /// <summary>
        /// The bottom Y coordinate of the word's bounding box.
        /// </summary>
        public double Bottom { get; init; }
    }
}
