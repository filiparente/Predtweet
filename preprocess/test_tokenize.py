from twokenize import tokenize
from pyTweetCleaner import TweetCleaner

print("\nOriginal sentence\n")
sentence = "@mia @bitcoin how u doing? ~135,67 hours to go! bag-of-words :O :) :P couldn't be more excited!!!! yes @ see you there /xoxo @keyvandavani @TuurDemeester @100trillionUSD @jimmysong @ToneVays @TraceMayer How many sats per person?\n ~3 million lost.\n 18 mil / 10 bil = 0.0018 BTC per person.\n = 180k sats per person\n Math can't answer the question though  because you would have to take hodlers of last resort into account. And other things  like timelocked and future lost coins etc. @maariana check this out! As Intel $INTC Valuation Rose , Btc Capital Management Has Increased by $418,968 Its Position ; Schlumberger LTD $SLB Share Price Declined While Mcdaniel Terry &amp; Co Cut by $1.42 Million Its Holding https://t.co/HmDE6GHvVW"
print(sentence)

print("\nSentence cleaned by pyTweetCleaner\n")
tc = TweetCleaner(remove_stop_words=False, remove_retweets=True)
print(tc.get_cleaned_text(sentence))

print("\nSentence cleaned by twokenize\n")
print(" ".join(tokenize(sentence)).encode('utf-8'))