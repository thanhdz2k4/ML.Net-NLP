using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ConsoleApp1
{
    public static class PorterStemmer
    {
        public static string Stem(string word)
        {
            if (word.Length < 3) return word;  // Words shorter than 3 characters are generally not stemmed.

            // Step 1a
            if (word.EndsWith("sses"))
            {
                word = word.Substring(0, word.Length - 2);
            }
            else if (word.EndsWith("ies"))
            {
                word = word.Substring(0, word.Length - 2);
            }
            else if (word.EndsWith("ss"))
            {
                // Do nothing, keep "ss" as it is
            }
            else if (word.EndsWith("s"))
            {
                word = word.Substring(0, word.Length - 1);
            }

            // Step 1b
            if (word.EndsWith("eed"))
            {
                if (Measure(word.Substring(0, word.Length - 3)) > 0)
                    word = word.Substring(0, word.Length - 1);
            }
            else if (word.EndsWith("ed") && ContainsVowel(word.Substring(0, word.Length - 2)))
            {
                word = word.Substring(0, word.Length - 2);
                word = Step1bHelper(word);
            }
            else if (word.EndsWith("ing") && ContainsVowel(word.Substring(0, word.Length - 3)))
            {
                word = word.Substring(0, word.Length - 3);
                word = Step1bHelper(word);
            }

            // Step 1c
            if (word.EndsWith("y") && ContainsVowel(word.Substring(0, word.Length - 1)))
            {
                word = word.Substring(0, word.Length - 1) + "i";
            }

            // You can add additional steps here for a more complete stemmer

            return word;
        }

        private static string Step1bHelper(string word)
        {
            if (word.EndsWith("at") || word.EndsWith("bl") || word.EndsWith("iz"))
            {
                word += "e";
            }
            else if (EndsWithDoubleConsonant(word) && !EndsWithAny(word, new[] { "l", "s", "z" }))
            {
                word = word.Substring(0, word.Length - 1);
            }
            else if (Measure(word) == 1 && EndsWithCVC(word))
            {
                word += "e";
            }

            return word;
        }

        private static bool ContainsVowel(string word)
        {
            return word.Any(c => "aeiou".Contains(c));
        }

        private static bool EndsWithDoubleConsonant(string word)
        {
            return word.Length > 1 && word[word.Length - 1] == word[word.Length - 2] &&
                   !"aeiou".Contains(word[word.Length - 1]);
        }

        private static bool EndsWithAny(string word, string[] suffixes)
        {
            return suffixes.Any(suffix => word.EndsWith(suffix));
        }

        private static bool EndsWithCVC(string word)
        {
            if (word.Length < 3) return false;
            char c = word[word.Length - 1];
            char v = word[word.Length - 2];
            char c2 = word[word.Length - 3];

            return !"aeiou".Contains(c) && "aeiou".Contains(v) && !"aeiouwxy".Contains(c2);
        }

        // Measures the "m" value in the Porter Stemmer algorithm
        private static int Measure(string word)
        {
            int m = 0;
            bool inVowelGroup = false;

            foreach (var c in word)
            {
                if ("aeiou".Contains(c))
                {
                    inVowelGroup = true;
                }
                else if (inVowelGroup)
                {
                    m++;
                    inVowelGroup = false;
                }
            }

            return m;
        }
    }

}
