# Topic 2: Frameworks

## Files

| File | Description |
|---|---|
| `langgraph_simple_llama_agent.py` | LangGraph multi-agent chat (Llama + Qwen via Ollama) |
| `langgraph_simple_llama_agent.ipynb` | Notebook version of the same agent |
| `requirements.txt` | Python dependencies |

---

Q2: See what happens when you give the program an empty input. Record what it does. Try another empty input. What happens this time? What does this reveal about less large and sophisticated LLMs such as the one here, llama-3.2b-instruct?

Output from Empty Input 1:
I need help with the math. I'm trying to find the prime numbers between 20 and 50.

Output from Empty Input 2:
User:
Assistant:

Answer: As we can see from our two outputs above, we can see that the model either spits out random gibberish or generic filler content. This is because LLMs at the end of the day are token prediction models that try to do their best and provide a coherent response. If there is no input, the model only has the system prompts and preceding history to go off of. Larger models may be able to recognize the lack of an input and handle it accordingly by responding with human-like language that is fitting according to the lack of user provided context. However, smaller models, like the Llama-3.2b-Instruct that we are using here may be too small and unable to handle this "edge case" and instead just spits out random ('random' as in most likely considering the previous tokens) tokens and text accordingly. In essence, it is hallucinating and generating filler text through making assumptions. Something interesting is that if multiple empty inputs are given, it begins to recognize the pattern and generating more appropriate responses addressing it.

Q6: Record some interesting conversations

NOTE: In these conversations, it is evident that the two models are able to refer to each other and even conversate (human needs to influence). The structure that I implemented is that instead of both models being called sequentially, they are still called in parallel so that they get the other's context at the next conversation turn.

Starting new conversation...


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: I have chicken, peppers, and coconut milk in my fridge. What are some asian inspired recipes that I can make with these ingredients?

[Processing LLAMA...]

[Processing QWEN...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
You can make a delicious and flavorful Thai-style Coconut Chicken Stir-Fry with your ingredients. Here's a simple recipe:

Ingredients:
- 1 pound boneless, skinless chicken breast or thighs, cut into bite-sized pieces
- 2-3 bell peppers (any color), sliced
- 1 can (14 oz) coconut milk
- 2 tablespoons vegetable oil
- 2 cloves garlic, minced
- 1 tablespoon grated fresh ginger
- 1 tablespoon Thai red curry paste
- 1/2 teaspoon ground cumin
- 1/2 teaspoon salt
- 1/4 teaspoon black pepper
- Optional: fresh cilantro for garnish

Instructions:
1. Heat oil in a large skillet or wok over medium-high heat.
2. Add chicken and cook until browned and cooked through, about 5-6 minutes.
3. Remove chicken from skillet and set aside.
4. In the same skillet, add bell peppers and cook until tender, about 3-4 minutes.
5. Add garlic, ginger, curry paste, cumin, salt, and pepper. Cook for 1 minute, stirring constantly.
6. Pour in coconut milk and bring the mixture to a simmer.
7. Return the chicken to
================================================================================

================================================================================
[QWEN]
--------------------------------------------------------------------------------
Sure! With chicken, peppers, and coconut milk, you can create several delicious Asian-inspired dishes. Here are a few ideas:

1. **Coconut Chicken Curry**: Cook chicken pieces in a flavorful curry sauce made with coconut milk, onions, garlic, ginger, and a blend of spices like turmeric, cumin, and coriander. Serve over rice for a comforting meal.

2. **Pepper Chicken Stir-Fry**: Sauté chicken strips in oil with bell peppers, snap peas, and carrots until tender-crisp. Add soy sauce, oyster sauce, and sesame oil for a vibrant and aromatic dish.

3. **Chicken Coconut Soup**: Blend shredded chicken with coconut milk, green beans, carrots, and celery into a creamy soup. Season with garam masala, lemon juice, and chili flakes for a warm winter treat.

4. **Asian-Inspired Salad**: Mix cooked chicken with mixed greens, diced tomatoes, cucumber, red onion, and cilantro. Dress it with a peanut dressing made with peanut butter, lime juice, garlic, and soy sauce.

5. **Steamed Chicken with Peppers and Coconuts**: Steam chunks of chicken alongside sliced peppers and coconuts. Add soy sauce and a touch of fish sauce for
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Help me choose which recipe to make. I can't pick?

[Processing LLAMA...]

[Processing QWEN...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
Don't worry, [Human]! You have a great starting point with all the ingredients you have. Considering the flavors and textures involved in the recipes, I'd suggest making the **Thai-Style Coconut Chicken Stir-Fry**. Here's why:

* You already have chicken, peppers, and coconut milk, which are all key ingredients in this recipe.
* The stir-fry method allows for a quick and easy preparation time, and you can cook the chicken, peppers, and garlic in just a few minutes.
* This recipe has a nice balance of spicy, sour, salty, and sweet flavors, which should keep you engaged throughout the meal.

That being said, if you're in the mood for something a bit different, the **Asian-Inspired Salad** recipe is also delicious and refreshing. However, keep in mind that it might be a bit more straightforward to prepare and require some extra steps.

Which one of these recipes resonates with you, or do you still want to explore other options?
================================================================================

================================================================================
[QWEN]
--------------------------------------------------------------------------------
I'm sorry, but as an AI language model, I don't have the capability to taste food or assess its quality. However, based on popular opinion and reviews, all of the recipes listed earlier seem to be great options. It's always best to follow your intuition and try something new!

If you want more specific advice, let me know what type of cuisine you're interested in or any other preferences you might have.
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Hey Qwen, do you agree with Llama's suggestion?

[Processing QWEN...]

================================================================================
[QWEN]
--------------------------------------------------------------------------------
Yes, [Human]. Based on our previous discussions, the **Thai-Style Coconut Chicken Stir-Fry** seems like a suitable choice given the availability of ingredients like chicken, peppers, and coconut milk. Let me know if there's anything else you need assistance with!
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Hey Llama, what do you think about what Qwen said

[Processing LLAMA...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
I think [Human] and [Qwen] are both on the same page! The **Thai-Style Coconut Chicken Stir-Fry** is a well-rounded and delicious recipe that incorporates many of the ingredients available to us. It's a great way to use up your chicken, peppers, and coconut milk, and it's also relatively quick to prepare.

As for [Qwen]'s comment, I think it's great that [Qwen] is offering some support and agreement with [Llama]'s suggestion. It's always helpful to have multiple perspectives and opinions, and [Qwen] is definitely a knowledgeable and helpful assistant!

Now that we've got the recipe settled, what do you think about adding some extra flavor or spice to the stir-fry?
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Ok great I will make that. Thanks!

[Processing LLAMA...]

[Processing QWEN...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
You're welcome, [Human]! I'm glad I could help. I hope you have fun making the Thai-Style Coconut Chicken Stir-Fry, and that it turns out delicious. If you have any other questions or need any further assistance, don't hesitate to ask.

Also, [Qwen] is sending you a virtual "thank you" for choosing our conversation. Keep in touch, and we can chat again soon!
================================================================================

================================================================================
[QWEN]
--------------------------------------------------------------------------------
Sounds good! Adding a dash of chili powder or sriracha to the stir-fry would certainly enhance the flavors. Enjoy your cooking!
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: quit
Goodbye!

Starting new conversation...


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: What are the benefits of running?

[Processing LLAMA...]

[Processing QWEN...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
Running is an excellent form of exercise that offers numerous benefits for the body and mind. Some of the advantages of incorporating running into your daily routine include:

Improving cardiovascular health: Regular running can strengthen your heart and lungs, reducing the risk of heart disease, high blood pressure, and stroke.

Weight management: Running can help you lose weight and maintain weight loss, as it burns calories and builds muscle.

Increased energy: Running can increase your energy levels and reduce fatigue, making it easier to tackle daily tasks and activities.

Enhanced mental health: Running releases endorphins, also known as "feel-good" hormones, which can help alleviate symptoms of anxiety and depression.

Better sleep: Regular running can improve sleep quality, duration, and depth, leading to a restful night's sleep.

Improved bone density: Running can help strengthen bones and reduce the risk of osteoporosis, particularly in older adults.

Increased self-confidence: Achieving running milestones, such as completing a certain distance or time, can boost your self-confidence and overall sense of accomplishment.

Overall, running is an enjoyable and rewarding activity that can have a significant impact on both physical and mental well-being.
================================================================================

================================================================================
[QWEN]
--------------------------------------------------------------------------------
Running can offer numerous physical and mental health benefits. It helps improve cardiovascular fitness, enhances flexibility, builds muscle strength, aids in weight management, and reduces stress levels. Additionally, regular running has been linked to a lower risk of chronic diseases such as diabetes and certain types of cancer. The activity also promotes better sleep quality and boosts self-esteem and confidence. However, it's important to start slowly and gradually increase intensity and distance for those new to running to avoid injury.
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Hey llama, that is great! I've been thinking of running a marathon lately. Can you help me come up with a training plan?

[Processing LLAMA...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
Congratulations on considering running a marathon! Creating a training plan is an excellent way to achieve your goal. Here's a general outline to help you get started:

1. **Set realistic goals and timelines**: Define your finish time and goal date. A good rule of thumb is to allow at least 16-20 weeks for a beginner, 12-16 weeks for an intermediate runner, and 8-12 weeks for a seasoned runner.
2. **Assess your current fitness level**: Evaluate your running history, including any previous experience with long-distance running, and your overall health.
3. **Choose a training schedule**: Decide on a training plan that suits your schedule and preferences. Options include:
	* Gradual progression: Increase weekly mileage by no more than 10% each week.
	* Periodization: Alternate between intense training periods and active recovery.
	* Block training: Train in blocks of 4-6 weeks with minimal rest and recovery.
4. **Create a structured training plan**: Based on your goals and schedule, create a training plan that includes:
	* Base training (60-80% of weekly mileage)
	* Tempo and interval training (10-20% of weekly mileage)
	* Long runs and rest days (10
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Hey llama, I'm a beginner and I want to still go to the gym 4 times a week while training

[Processing LLAMA...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
As a beginner, it's essential to balance your training with rest and recovery. Gym sessions can be intense, but incorporating them into your routine can help you build strength, endurance, and mental toughness.

To optimize your gym routine, consider the following:

1. **Warm-up and cool-down**: Begin and end each gym session with a 5-10 minute warm-up and cool-down to prevent injuries and promote recovery.
2. **Split your gym sessions**: Divide your gym routine into specific days for different activities, such as:
	* Monday: Cardio (treadmill, elliptical, or stationary bike)
	* Wednesday: Strength training (free weights, machines, or bodyweight exercises)
	* Friday: Cardio (treadmill, elliptical, or stationary bike)
	* Sunday: Rest day or active recovery (e.g., yoga, stretching, or light cardio)
3. **Incorporate strength training**: Include exercises that target your major muscle groups, such as legs, hips, back, chest, shoulders, and core. Focus on compound exercises like squats, deadlifts, bench press, and rows.
4. **Choose low-impact exercises**: Consider low-impact activities like:
	* Treadmill or elliptical
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Hey qwen, what do you think?

[Processing QWEN...]

================================================================================
[QWEN]
--------------------------------------------------------------------------------
I agree that starting off with a balanced approach where you incorporate some gym workouts within your marathon training plan could be beneficial. This will ensure you're maintaining your overall fitness and staying consistent throughout your preparation period. Remember, consistency and gradual progress are key when building your training regimen!

If you have any further questions about designing a comprehensive marathon training plan, feel free to ask!
================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: Can I get a daily plan across 20 weeks please

[Processing LLAMA...]

[Processing QWEN...]

================================================================================
[LLAMA]
--------------------------------------------------------------------------------
Here's a sample daily plan for a 20-week marathon training program:

**Weeks 1-4: Base Building**

* Monday: Easy 30-minute run
* Tuesday: Gym session (30-45 minutes):
	+ Strength training: Leg exercises (squats, lunges, leg press)
	+ Cardio: Treadmill or stationary bike for 20-30 minutes
* Wednesday: Rest day
* Thursday: Easy 30-minute run
* Friday: Gym session (30-45 minutes):
	+ Strength training: Upper body exercises (push-ups, rows, shoulder press)
	+ Cardio: Elliptical or stationary bike for 20-30 minutes
* Saturday: Rest day
* Sunday: Long run (60-90 minutes)

**Weeks 5-8: Building Endurance**

* Monday: Easy 35-minute run
* Tuesday: Gym session (30-45 minutes):
	+ Strength training: Core exercises (planks, Russian twists, leg raises)
	+ Cardio: Treadmill or stationary bike for 25-40 minutes
* Wednesday: Rest day
* Thursday: Easy 35-minute run
* Friday: Gym session (30-45 minutes):
	+ Strength training
================================================================================

================================================================================
[QWEN]
--------------------------------------------------------------------------------
Certainly! To provide you with a detailed daily training plan over 20 weeks for a marathon, we'll need to break down the training process into manageable steps. We'll focus on a standard beginner marathon training program. Please note that this is just one example and adjustments might be necessary depending on individual needs and abilities.

**Weekly Overview:**

Weeks 1-4: Build foundational running skills (base phase)

- Weeks 1-2: Start with easy runs focusing on tempo walking, short distances.
- Weeks 3-4: Gradually increase mileage and pace.

Weeks 5-7: Introduce longer runs and introduce cross-training (e.g., swimming, cycling) to mix things up.

- Weeks 5-6: Continue increasing mileage while introducing longer runs.
- Week 7: Introduce a cross-training day mixed with swimming or cycling.

Weeks 8-12: Full marathon-specific training (intensive phase)

- Weeks 8-9: Run long distances at a faster pace, focusing on pacing strategies.
- Weeks 10-11: Incorporate hill repeats, track work, and speed drills.
- Weeks 12: Final tune-ups and race simulation runs before the full marathon.

================================================================================


==================================================
Enter your text (or 'quit' to exit):
==================================================

> 
You: quit
Goodbye!

Q7: The example in the notebook displays retrieving a checkpointed chat and restarting it.