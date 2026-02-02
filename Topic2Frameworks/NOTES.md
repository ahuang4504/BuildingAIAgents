Q2: See what happens when you give the program an empty input. Record what it does. Try another empty input. What happens this time? What does this reveal about less large and sophisticated LLMs such as the one here, llama-3.2b-instruct?

Output from Empty Input 1:
I need help with the math. I'm trying to find the prime numbers between 20 and 50.

Output from Empty Input 2:
User:
Assistant:

Answer: As we can see from our two outputs above, we can see that the model either spits out random gibberish or generic filler content. This is because LLMs at the end of the day are token prediction models that try to do their best and provide a coherent response. If there is no input, the model only has the system prompts and preceding history to go off of. Larger models may be able to recognize the lack of an input and handle it accordingly by responding with human-like language that is fitting according to the lack of user provided context. However, smaller models, like the Llama-3.2b-Instruct that we are using here may be too small and unable to handle this "edge case" and instead just spits out random ('random' as in most likely considering the previous tokens) tokens and text accordingly. In essence, it is hallucinating and generating filler text through making assumptions. Something interesting is that if multiple empty inputs are given, it begins to recognize the pattern and generating more appropriate responses addressing it.
