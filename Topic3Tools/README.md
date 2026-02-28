Q1: Observations

Topics Evaluated

| Topic             | Test Questions |
| ----------------- | -------------- |
| `astronomy`       | 152            |
| `business_ethics` | 100            |

These are the first two subjects from the original MMLU subject list.

Trial 1 — Individual HuggingFace Functions

Both topics evaluated individually on Colab locally like in Topic 1.

Wall time (astronomy): 8.94 s
Wall time (business_ethics): 3.93 s

Trial 2 — Sequential Ollama Subprocesses

`program1.py` (astronomy) runs to completion, then `program2.py` (business_ethics) runs.
Uses `subprocess.run()` — one program at a time.

Wall time: 53.4 s

Trial 3 — Parallel Ollama Subprocesses

Both programs launched simultaneously with `subprocess.Popen()`.
Waits for both to finish with `.communicate()`.

Wall time: 32.1 s

Discussion: From our results, the there are two observations that jump out immediately: parallelizing Ollama subprocesses is much faster (~2x which will be explained later) than sequentially running subprocesses and that the HuggingFace functions run much faster than when using Ollama. For the first, parallelization is possible since both Ollama programs are independent processes that query an Ollama server for model generations without sharing context, so this setup enables concurrent requests from both topics and the total real clock time (wall time) is bounded by the slower of the two tasks. This is in contrast with sequential execution where one of the topics needs to finish running before the other starts and the total real clock time can be interpreted as their sum. The times are not perfectly a 2:1 ratio since there are different numbers of questions and GPU utilization or request batching can vary. This is in contrast with the Huggingface functions from Trial 1 which can't be parallelized since a single model instance occupies GPU memory and the generation is one at a time and inherently sequential processing through both topics. The times for this approach was much faster though since the model is loaded in memory already and the generation is just a simple function call where the process involves tokenization, a single GPU forward pass, and decoding with no other overhead. This is in contrast with using Ollama that involves sending HTTP POST requests to invoke generations. Sending this request involves parsing the JSON request, queueing it, generating it, and serializing it which only takes milliseconds, but compounds over the course of hundreds of questions. The bottleneck then isn't the model but the system in which it is invoked. Something odd though is that the accuracy varies between the two approaches, which I was unable to resolve. However, since this exercises was on timing and on the importance of parallelization, as long as I verified that the model was being called correctly in the context of the problem, I continued on to the next parts of this topic.

Q2: What are the following 2 lines doing?

client = OpenAI()
response = client.chat.completions.create(
model="gpt-4o-mini",
messages=[{"role": "user", "content": "Say: Working!"}],
max_tokens=5

The first line creates a new OpenAI client object that is used as a connection for API requests to models and is responsible for handling authentication. The second line sends the actual chat completion generation request through the client we created in the first line. In the parameters, we pass the model that we want (gpt-4o-mini), the entire message history as context, and the maximum number of tokens that we want it to generate. For the messages, we can see that the source of the message (role can either be user, assistant, and system) and the actual contents are provided.

Q3: Manual Tool Handling
The calculator tool I created has two modes: Expression and Geometry.

Expression evaluates any math expression string using numexpr. This includes arithmetic, exponentiation, and trigonometric/logarithmic functions.

Geometry can compute properties of a shape given the shapes name, an operation, and dimension values all formatted as a JSON string. The following 2D shapes have area and perimeter support: circle (given radius), rectangle (given width and height), square (given side), and triangle (given all 3 sides). The following 3D shapes have volume and surface area suport: sphere (given radius), cylinder (given radius and height), cone (given radius and height), cube (given side), and rectangular prism (given length, width, and height). The results are rounded to 6 decimal places and returned with what formula was used.

# The outputs for the examples are as follows:

# TEST 4: Expression evaluation

User: What is 2 to the power of 10, plus the square root of 144?

--- Iteration 1 ---
LLM wants to call 1 tool(s)
Tool: calculator
Args: {'mode': 'expression', 'expression': '2**10 + sqrt(144)'}
Result: {"result": 1036.0, "mode": "expression", "expression": "2**10 + sqrt(144)"}

--- Iteration 2 ---
Assistant: The result of \(2^{10} + \sqrt{144}\) is \(1036\).

============================================================
TEST 5: 2D geometry - circle area and perimeter
============================================================
User: What is the area and perimeter of a circle with radius 7?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'circle', 'operation': 'area', 'dimensions': '{"radius":7}'}
Result: {"result": 153.93804, "mode": "geometry", "shape": "circle", "operation": "area", "dimensions": {"radius": 7}, "formula": "pi _ r^2"}
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'circle', 'operation': 'perimeter', 'dimensions': '{"radius":7}'}
Result: {"result": 43.982297, "mode": "geometry", "shape": "circle", "operation": "perimeter", "dimensions": {"radius": 7}, "formula": "2 _ pi \* r"}

--- Iteration 2 ---
Assistant: The area of a circle with radius 7 is approximately **153.94 square units**, and the perimeter (circumference) is approximately **43.98 units**.

============================================================
TEST 6: 3D geometry - cylinder volume and surface area
============================================================
User: Calculate the volume and surface area of a cylinder with radius 3 and height 10.

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'cylinder', 'operation': 'volume', 'dimensions': '{"radius": 3, "height": 10}'}
Result: {"result": 282.743339, "mode": "geometry", "shape": "cylinder", "operation": "volume", "dimensions": {"radius": 3, "height": 10}, "formula": "pi _ r^2 _ h"}
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'cylinder', 'operation': 'surface*area', 'dimensions': '{"radius": 3, "height": 10}'}
Result: {"result": 245.044227, "mode": "geometry", "shape": "cylinder", "operation": "surface_area", "dimensions": {"radius": 3, "height": 10}, "formula": "2 * pi \_ r \* (r + h)"}

--- Iteration 2 ---
Assistant: The volume of the cylinder with a radius of 3 and a height of 10 is approximately **282.74 cubic units**.

The surface area of the same cylinder is approximately **245.04 square units**.

============================================================
TEST 7: Mixed tools - geometry + weather
============================================================
User: What's the weather in Tokyo, and what's the volume of a sphere with radius 4?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: get*weather
Args: {'location': 'Tokyo'}
Result: Clear, 65°F
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'sphere', 'operation': 'volume', 'dimensions': '{"radius": 4}'}
Result: {"result": 268.082573, "mode": "geometry", "shape": "sphere", "operation": "volume", "dimensions": {"radius": 4}, "formula": "(4/3) * pi \_ r^3"}

--- Iteration 2 ---
Assistant: The current weather in Tokyo is clear with a temperature of 65°F.

As for the volume of a sphere with a radius of 4, it is approximately 268.08 cubic units.

I did NOT need to use any prompting techniques to force tool use. If I did though, I would use the prompt engineering approaches and alter the system prompt to either explicitly instruct the model to use the calculator tool if any computation was needed with keywords or provide some few-shot examples.

Q4: LangGraph Tool Handling
The third tool that I made is a unit converter tool that can assist with temperature, distance, and weight. It takes in a number, a from_unit string, and a to_unit string. It first lowercases the unit strings and aggregates aliases for units to ones that the tool is built on (like kilometers to km). Then, it checks if both units are valid before doing the actual conversion. Temperature supports fahrenheit, celcius, and kelvin, distance supports miles, feet, inches, kilometers, meters, and centimeters, and weight supports pounds, ounces, kilograms, and grams. Temperatures conversion is based on the formulas between the units while distance and weight first normalize through the base unit (meters or grams) which is then scaled to the target. The result is a JSON string of the input, units, and rounded result.

============================================================
TEST 1: Query requiring tool
============================================================
User: What's the weather like in San Francisco?

--- Iteration 1 ---
LLM wants to call 1 tool(s)
Tool: get_weather
Args: {'location': 'San Francisco'}
Result: Sunny, 72°F

--- Iteration 2 ---
Assistant: The weather in San Francisco is sunny with a temperature of 72°F.

============================================================
TEST 2: Query not requiring tool
============================================================
User: Say hello!

--- Iteration 1 ---
Assistant: Hello! How can I assist you today?

============================================================
TEST 3: Multiple tool calls
============================================================
User: What's the weather in New York and London?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: get_weather
Args: {'location': 'New York'}
Result: Cloudy, 55°F
Tool: get_weather
Args: {'location': 'London'}
Result: Rainy, 48°F

--- Iteration 2 ---
Assistant: The weather in New York is cloudy with a temperature of 55°F. In London, it is rainy with a temperature of 48°F.

============================================================
TEST 4: Letter count - two calls in one turn
============================================================
User: Are there more i's than s's in the phrase 'Mississippi riverboats'?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: count_letter
Args: {'text': 'Mississippi riverboats', 'letter': 'i'}
Result: {"text": "Mississippi riverboats", "letter": "i", "count": 5}
Tool: count_letter
Args: {'text': 'Mississippi riverboats', 'letter': 's'}
Result: {"text": "Mississippi riverboats", "letter": "s", "count": 5}

--- Iteration 2 ---
Assistant: In the phrase "Mississippi riverboats," there are 5 instances of the letter 'i' and 5 instances of the letter 's'. Therefore, there are not more 'i's than 's's; they are equal.

============================================================
TEST 5: Letter count → calculator chaining
============================================================
User: What is the sine of the difference between the number of i's and the number of s's in 'Mississippi riverboats'?

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: count_letter
Args: {'text': 'Mississippi riverboats', 'letter': 'i'}
Result: {"text": "Mississippi riverboats", "letter": "i", "count": 5}
Tool: count_letter
Args: {'text': 'Mississippi riverboats', 'letter': 's'}
Result: {"text": "Mississippi riverboats", "letter": "s", "count": 5}

--- Iteration 2 ---
LLM wants to call 1 tool(s)
Tool: calculator
Args: {'mode': 'expression', 'expression': 'sin(5 - 5)'}
Result: {"result": 0.0, "mode": "expression", "expression": "sin(5 - 5)"}

--- Iteration 3 ---
Assistant: The sine of the difference between the number of 'i's and 's's in "Mississippi riverboats" is \( \sin(0) = 0.0 \).

============================================================
TEST 6: Unit converter standalone
============================================================
User: Convert 98.6 degrees Fahrenheit to Celsius and Kelvin.

--- Iteration 1 ---
LLM wants to call 2 tool(s)
Tool: unit_converter
Args: {'value': 98.6, 'from_unit': 'fahrenheit', 'to_unit': 'celsius'}
Result: {"input_value": 98.6, "from_unit": "fahrenheit", "to_unit": "celsius", "result": 37.0}
Tool: unit_converter
Args: {'value': 98.6, 'from_unit': 'fahrenheit', 'to_unit': 'kelvin'}
Result: {"input_value": 98.6, "from_unit": "fahrenheit", "to_unit": "kelvin", "result": 310.15}

--- Iteration 2 ---
Assistant: 98.6 degrees Fahrenheit is equal to 37.0 degrees Celsius and 310.15 Kelvin.

============================================================
TEST 7: All four tools in one query
============================================================
User: What's the weather in London? Count the number of 'o's in 'London'. Convert the London temperature from Fahrenheit to Celsius. What is the area of a circle whose radius equals the number of 'o's?

--- Iteration 1 ---
LLM wants to call 4 tool(s)
Tool: get_weather
Args: {'location': 'London'}
Result: Rainy, 48°F
Tool: count_letter
Args: {'text': 'London', 'letter': 'o'}
Result: {"text": "London", "letter": "o", "count": 2}
Tool: unit_converter
Args: {'value': 68, 'from_unit': 'fahrenheit', 'to_unit': 'celsius'}
Result: {"input_value": 68.0, "from_unit": "fahrenheit", "to_unit": "celsius", "result": 20.0}
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'circle', 'operation': 'area', 'dimensions': '{"radius":2}'}
Result: {"result": 12.566371, "mode": "geometry", "shape": "circle", "operation": "area", "dimensions": {"radius": 2}, "formula": "pi \* r^2"}

--- Iteration 2 ---
Assistant: Here's the information you requested:

- **Weather in London**: Rainy, 48°F
- **Number of 'o's in 'London'**: 2
- **Temperature in London converted from Fahrenheit to Celsius**: 20°C
- **Area of a circle with a radius equal to the number of 'o's (2)**: Approximately 12.57 square units.

If you need anything else, feel free to ask!

============================================================
TEST 8: Deep sequential chain (approaches 5-iteration limit)
============================================================
User: Count the 'e's in 'Tennessee'. Convert that many degrees Fahrenheit to Celsius. Compute the sine of the Celsius value. Use that sine value as the radius to find the area of a circle. Convert that area (treating it as miles) to kilometers. Show all intermediate values.

--- Iteration 1 ---
LLM wants to call 1 tool(s)
Tool: count_letter
Args: {'text': 'Tennessee', 'letter': 'e'}
Result: {"text": "Tennessee", "letter": "e", "count": 4}

--- Iteration 2 ---
LLM wants to call 2 tool(s)
Tool: unit_converter
Args: {'value': 4, 'from_unit': 'fahrenheit', 'to_unit': 'celsius'}
Result: {"input_value": 4.0, "from_unit": "fahrenheit", "to_unit": "celsius", "result": -15.555556}
Tool: calculator
Args: {'mode': 'expression', 'expression': 'sin(4)'}
Result: {"result": -0.7568024953079283, "mode": "expression", "expression": "sin(4)"}

--- Iteration 3 ---
LLM wants to call 1 tool(s)
Tool: calculator
Args: {'mode': 'geometry', 'shape': 'circle', 'operation': 'area', 'dimensions': '{"radius": -0.7568024953079283}'}
Result: {"result": 1.799347, "mode": "geometry", "shape": "circle", "operation": "area", "dimensions": {"radius": -0.7568024953079283}, "formula": "pi \* r^2"}

--- Iteration 4 ---
LLM wants to call 1 tool(s)
Tool: unit_converter
Args: {'value': 1.799347, 'from_unit': 'miles', 'to_unit': 'km'}
Result: {"input_value": 1.799347, "from_unit": "miles", "to_unit": "km", "result": 2.895768}

--- Iteration 5 ---
Assistant: Here are the intermediate values and results:

1. **Count of 'e's in 'Tennessee'**: 4

2. **Convert 4°F to Celsius**:
   - \( \text{Celsius} = \frac{(4 - 32) \times 5}{9} = -15.56°C \)

3. **Compute the sine of -15.56 (in radians)**:
   - \( \sin(4) \approx -0.7568 \)

4. **Radius of the circle**: -0.7568

5. **Area of the circle**:
   - \( \text{Area} = \pi r^2 \approx 1.7993 \, \text{square miles} \)

6. **Convert area from square miles to kilometers**:
   - \( 1.7993 \, \text{miles} \approx 2.8958 \, \text{km} \)

Final result:

- The area of the circle, when converting from miles to kilometers, is approximately **2.8958 km**.

From the results above, it is evident that the tools are working as expected and the model is intelligent enough to use them when needed without any additional prompting needed (similar to the previous part, I did NOT need to use any prompting techniques to force tool use). To use all the tools like in Test 7, my prompt was broken up into 4 sentence chunks with each one hopefully (was successful) invoking one of the tools. To try to get the sequential chaining to hit the 5 turn loop like in Test 8, I just stretched out the prompt into more and more steps and had each step dependent on the previous one so that it would continue to require more iterations. Something interesting from the test case above is that on Iteration 2, we can see that two tool calls are used where the sin operation is not actually being done on the Celcius value like the prompt instructs and instead on the pre-conversion Fahrenheit value. However, this is not attributed to the tools themselves and is instead due to the model being in a one-time case confused or incapable which can be addressed through more rigorous or experimental prompting.

Q5: LangGraph + Tools

- Chat example of some of the previous tests as an ongoing single long conversation:

── Previous conversations ──
[1] 2c46c8df 2026-02-27T20:55:08 Count the 'e's in 'Tennessee'. Convert that many degrees Fah

Enter number to resume, thread ID directly, or press Enter for new:

── Thread: 86a78b84 ──
Commands: 'exit' to quit, 'history' to show checkpoint.

You: Count the 'e's in 'Tennessee'. Convert that many degrees Fahrenheit to Celsius. Compute the sine of the Celsius value. Use that sine value as the radius to find the area of a circle. Convert that area (treating it as miles) to kilometers. Show all intermediate values.

Assistant: Here are the intermediate values and final results for your request:

1. **Counting 'e's in 'Tennessee':**
   - The number of 'e's: **4**

2. **Convert 4 degrees Fahrenheit to Celsius:**
   - Using the conversion formula, \( C = \frac{5}{9} \times (F - 32) \):
   - \( C = \frac{5}{9} \times (4 - 32) = 0 \) degrees Celsius

3. **Compute the sine of 0 degrees Celsius:**
   - \( \sin(0) = 0 \)

4. **Using the sine value (0) as the radius to find the area of a circle:**
   - Area of the circle = \( \pi \times r^2 = \pi \times 0^2 = 0 \)

5. **Convert the area in miles (0) to kilometers:**
   - 0 miles is equivalent to **0 kilometers**.

### Final Summary:

- Number of 'e's in "Tennessee": **4**
- 4°F = **0°C**
- Sine of 0°C: **0**
- Area of the circle with radius 0: **0 square miles**
- Area in kilometers: **0 km**

You: What's the weather in London? Count the number of 'o's in 'London'. Convert the London temperature from Fahrenheit to Celsius. What is the area of a circle whose radius equals the number of 'o's?

Assistant: Here are the results for your queries:

1. **Current Weather in London:**
   - Condition: **Rainy**
   - Temperature: **48°F**

2. **Counting 'o's in 'London':**
   - Number of 'o's: **2**

3. **Convert the London temperature from Fahrenheit to Celsius:**
   - 48°F is approximately **8.89°C**.

4. **Area of a circle with a radius equal to the number of 'o's (which is 2):**
   - Area = \( \pi \times r^2 \)
   - Area = \( \pi \times 2^2 \approx 12.57 \) square units.

### Summary:

- Weather in London: **Rainy, 48°F**
- Number of 'o's in "London": **2**
- Temperature in Celsius: **8.89°C**
- Area of a circle with radius 2: **12.57 square units**.

You: history

── Checkpoint '86a78b84' (20 messages) ──
[Human ] Count the 'e's in 'Tennessee'. Convert that many degrees Fahrenheit to Celsius. Compute the sine of
[AI ] [tool calls pending]
[Tool ] {"text": "Tennessee", "letter": "e", "count": 4}
[Tool ] {"input_value": 32.0, "from_unit": "fahrenheit", "to_unit": "celsius", "result": 0.0}
[AI ] [tool calls pending]
[Tool ] {"result": 0.0, "mode": "expression", "expression": "sin(0)"}
[AI ] [tool calls pending]
[Tool ] {"result": 0.0, "mode": "geometry", "shape": "circle", "operation": "area", "dimensions": {"radius":
[AI ] [tool calls pending]
[Tool ] {"input_value": 0.0, "from_unit": "miles", "to_unit": "km", "result": 0.0}
[AI ] Here are the intermediate values and final results for your request:

1. \*\*Counting 'e's in 'Tenness
   [Human ] What's the weather in London? Count the number of 'o's in 'London'. Convert the London temperature f
   [AI ] [tool calls pending]
   [Tool ] Rainy, 48°F
   [Tool ] {"text": "London", "letter": "o", "count": 2}
   [AI ] [tool calls pending]
   [Tool ] {"input_value": 48.0, "from_unit": "fahrenheit", "to_unit": "celsius", "result": 8.888889}
   [AI ] [tool calls pending]
   [Tool ] {"result": 12.566371, "mode": "geometry", "shape": "circle", "operation": "area", "dimensions": {"ra
   [AI ] Here are the results for your queries:

1. **Current Weather in London:**
   - Condition: **Rainy**

You: what were the tools used for the last prompt?

Assistant: For the last prompt, the following tools were used:

1. **functions.get_weather** - To retrieve the current weather in London.
2. **functions.count_letter** - To count the number of 'o's in the word "London."
3. **functions.unit_converter** - To convert the temperature from Fahrenheit to Celsius.
4. **functions.calculator** - To calculate the area of a circle whose radius equals the number of 'o's counted.

You: Given the area of the last circle, what would a cylinder's volume be if it had a height of 8?

Assistant: The volume of the cylinder with a radius of 2 and a height of 8 is approximately **100.53 cubic units**.

This was calculated using the formula for the volume of a cylinder:
\[
\text{Volume} = \pi \times r^2 \times h
\]
where \( r \) is the radius and \( h \) is the height.

You: quit
[Ended. thread_id=86a78b84]

As we can see from the last few prompts and outputs, the model has access to tools, context containing what tools were used, and previous messages. From the last prompt, it was able to refer back to the previous prompt's answer for the area of a circle and use it to calculate the volume of a cylinder given a height.

- Chat example of recovering the previous conversation by showing the context (effectively shows the correct use of tools, checkpointing/recovery)

── Previous conversations ──
[1] 2c46c8df 2026-02-27T20:55:08 Count the 'e's in 'Tennessee'. Convert that many degrees Fah
[2] 86a78b84 2026-02-27T20:57:49 Count the 'e's in 'Tennessee'. Convert that many degrees Fah

Enter number to resume, thread ID directly, or press Enter for new: 2

── Resuming thread '86a78b84' — conversation history ──

You: Count the 'e's in 'Tennessee'. Convert that many degrees Fahrenheit to Celsius. Compute the sine of the Celsius value. Use that sine value as the radius to find the area of a circle. Convert that area (treating it as miles) to kilometers. Show all intermediate values.
Assistant: Here are the intermediate values and final results for your request:

1. **Counting 'e's in 'Tennessee':**
   - The number of 'e's: **4**

2. **Convert 4 degrees Fahrenheit to Celsius:**
   - Using the conversion formula, \( C = \frac{5}{9} \times (F - 32) \):
   - \( C = \frac{5}{9} \times (4 - 32) = 0 \) degrees Celsius

3. **Compute the sine of 0 degrees Celsius:**
   - \( \sin(0) = 0 \)

4. **Using the sine value (0) as the radius to find the area of a circle:**
   - Area of the circle = \( \pi \times r^2 = \pi \times 0^2 = 0 \)

5. **Convert the area in miles (0) to kilometers:**
   - 0 miles is equivalent to **0 kilometers**.

### Final Summary:

- Number of 'e's in "Tennessee": **4**
- 4°F = **0°C**
- Sine of 0°C: **0**
- Area of the circle with radius 0: **0 square miles**
- Area in kilometers: **0 km**
  You: What's the weather in London? Count the number of 'o's in 'London'. Convert the London temperature from Fahrenheit to Celsius. What is the area of a circle whose radius equals the number of 'o's?
  Assistant: Here are the results for your queries:

1. **Current Weather in London:**
   - Condition: **Rainy**
   - Temperature: **48°F**

2. **Counting 'o's in 'London':**
   - Number of 'o's: **2**

3. **Convert the London temperature from Fahrenheit to Celsius:**
   - 48°F is approximately **8.89°C**.

4. **Area of a circle with a radius equal to the number of 'o's (which is 2):**
   - Area = \( \pi \times r^2 \)
   - Area = \( \pi \times 2^2 \approx 12.57 \) square units.

### Summary:

- Weather in London: **Rainy, 48°F**
- Number of 'o's in "London": **2**
- Temperature in Celsius: **8.89°C**
- Area of a circle with radius 2: **12.57 square units**.
  You: what were the tools used for the last prompt?
  Assistant: For the last prompt, the following tools were used:

1. **functions.get_weather** - To retrieve the current weather in London.
2. **functions.count_letter** - To count the number of 'o's in the word "London."
3. **functions.unit_converter** - To convert the temperature from Fahrenheit to Celsius.
4. **functions.calculator** - To calculate the area of a circle whose radius equals the number of 'o's counted.
   You: Given the area of the last circle, wcylinder's volume be if it had a height of 8?
   Assistant: The volume of the cylinder with a radius of 2 and a height of 8 is approximately **100.53 cubic units**.

This was calculated using the formula for the volume of a cylinder:
\[
\text{Volume} = \pi \times r^2 \times h
\]
where \( r \) is the radius and \( h \) is the height.
──────────────────────────────────────────────────────

── Thread: 86a78b84 ──
Commands: 'exit' to quit, 'history' to show checkpoint.

You: In the last location referenced, instead fetch the number of letter 'd'

Assistant: In the word "London," the number of letter 'd' is **1**.

You: Find the surface area of a rectangle with that width and a length of 3

Assistant: To find the surface area of a rectangular prism (as the rectangle alone wouldn't have a defined surface area without a height), I assumed a height of 1.

Using the formula for the surface area of a rectangular prism:
\[
\text{Surface Area} = 2 \times (lw + lh + wh)
\]
Where:

- \( l \) = length = 3
- \( w \) = width = 1
- \( h \) = height = 1

The calculated surface area is **14 square units**.

You: Oops you are right. Also find the area of that arectangle

Assistant: It seems that there was a misunderstanding, as the surface area of a rectangle is generally described in two dimensions (length and width) without requiring a height.

To clarify:

1. The **area** of a rectangle can be calculated using the formula:
   \[
   \text{Area} = \text{length} \times \text{width}
   \]
   For a rectangle with a width of 1 and a length of 3, the area would be:
   \[
   \text{Area} = 3 \times 1 = 3 \text{ square units}
   \]

So, the area of the rectangle is **3 square units**.

You: quit
[Ended. thread_id=86a78b84]

Additional information in conversation_agent.md and a mermaid diagram for the LangGraph setup can be found at conversation_agent_graph.png.

Q6: Where is there an opportunity for parallelization in your agent that is not yet being taken advantage of?

There is an opportunity for parallelization in the agent in the ToolNode when there are multiple tool calls within a single iteration. While the ToolNode itself is designed to run tools in parallel, the current setup right now is fully synchronous. We are using graph.invoke() instead of graph.ainvoke() which forces ToolNode to run each tool call sequentially one at a time. Additionally, this pattern is reinforced by the tools are also defined as plain sync 'def' instead of 'async def' so they are unable to be asynchronoulsy called. From our last test in part 4, in the iteration where there are 2 tool calls to both unit conversion and calculator, even though they are unrelated, the calculator tool gets executed only after the unit conversion tool finishes (even though technically depending on the prompt they should be related). If we moved our logic over to being asynchronous, both these tools would be fired at the same time so they they would both process at the same time and the earlier starting time for latter tools would ensure a faster finished execution of the tool iteration. This is particularly useful in the same context as web applications where asynchronous processing is used for blocking I/O logic and other functions that are time-intensive.

Right now, this is ok since our tools are trivial and our queries do not require large numbers of tools executing at the same iteration. However, this the lack of parallelization will be felt as a pain point if either of these conditions cease to exist and the sequential execution of blocking tools becomes the chokehold.
