"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

import json
import math
import os
from typing import Optional
import numexpr
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

load_dotenv()

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculator(
    mode: str,
    expression: Optional[str] = None,
    shape: Optional[str] = None,
    dimensions: Optional[str] = None,
    operation: Optional[str] = None,
) -> str:
    """Multi-purpose calculator. Use mode='expression' to evaluate a math string
    (e.g. '2**10 + sqrt(144)', 'sin(3.14159)'). Use mode='geometry' with shape,
    operation, and dimensions (JSON string) to compute area, perimeter, volume,
    or surface_area for: circle, rectangle, triangle, square, sphere, cylinder,
    cone, cube, rectangular_prism."""
    try:
        if mode == "expression":
            if expression is None:
                return json.dumps({"error": "expression is required in expression mode"})
            result = float(numexpr.evaluate(expression))
            return json.dumps({"result": result, "mode": "expression", "expression": expression})

        elif mode == "geometry":
            if not all([shape, dimensions, operation]):
                return json.dumps({"error": "shape, dimensions, and operation are required in geometry mode"})

            dims = json.loads(dimensions)
            shape, operation = shape.lower(), operation.lower()
            result, formula = None, ""

            if shape == "circle":
                r = dims["radius"]
                if operation == "area":        result, formula = math.pi * r**2,       "pi * r^2"
                elif operation == "perimeter": result, formula = 2 * math.pi * r,      "2 * pi * r"
                else: return json.dumps({"error": f"circle doesn't support '{operation}'"})

            elif shape == "rectangle":
                w, h = dims["width"], dims["height"]
                if operation == "area":        result, formula = w * h,                "width * height"
                elif operation == "perimeter": result, formula = 2 * (w + h),          "2 * (width + height)"
                else: return json.dumps({"error": f"rectangle doesn't support '{operation}'"})

            elif shape == "triangle":
                a, b, c = dims["a"], dims["b"], dims["c"]
                if operation == "area":
                    s = (a + b + c) / 2
                    result, formula = math.sqrt(s*(s-a)*(s-b)*(s-c)), "Heron's formula"
                elif operation == "perimeter": result, formula = a + b + c, "a + b + c"
                else: return json.dumps({"error": f"triangle doesn't support '{operation}'"})

            elif shape == "square":
                s = dims["side"]
                if operation == "area":        result, formula = s**2,   "side^2"
                elif operation == "perimeter": result, formula = 4 * s,  "4 * side"
                else: return json.dumps({"error": f"square doesn't support '{operation}'"})

            elif shape == "sphere":
                r = dims["radius"]
                if operation == "volume":         result, formula = (4/3) * math.pi * r**3, "(4/3) * pi * r^3"
                elif operation == "surface_area": result, formula = 4 * math.pi * r**2,    "4 * pi * r^2"
                else: return json.dumps({"error": f"sphere doesn't support '{operation}'"})

            elif shape == "cylinder":
                r, h = dims["radius"], dims["height"]
                if operation == "volume":         result, formula = math.pi * r**2 * h,        "pi * r^2 * h"
                elif operation == "surface_area": result, formula = 2 * math.pi * r * (r + h), "2 * pi * r * (r + h)"
                else: return json.dumps({"error": f"cylinder doesn't support '{operation}'"})

            elif shape == "cone":
                r, h = dims["radius"], dims["height"]
                slant = math.sqrt(r**2 + h**2)
                if operation == "volume":         result, formula = (1/3) * math.pi * r**2 * h,  "(1/3) * pi * r^2 * h"
                elif operation == "surface_area": result, formula = math.pi * r * (r + slant),   "pi * r * (r + slant)"
                else: return json.dumps({"error": f"cone doesn't support '{operation}'"})

            elif shape == "cube":
                s = dims["side"]
                if operation == "volume":         result, formula = s**3,      "side^3"
                elif operation == "surface_area": result, formula = 6 * s**2,  "6 * side^2"
                else: return json.dumps({"error": f"cube doesn't support '{operation}'"})

            elif shape == "rectangular_prism":
                l, w, h = dims["length"], dims["width"], dims["height"]
                if operation == "volume":         result, formula = l * w * h,             "length * width * height"
                elif operation == "surface_area": result, formula = 2*(l*w + l*h + w*h),  "2*(lw + lh + wh)"
                else: return json.dumps({"error": f"rectangular_prism doesn't support '{operation}'"})

            else:
                return json.dumps({"error": f"Unknown shape: '{shape}'"})

            return json.dumps({
                "result": round(result, 6),
                "mode": "geometry",
                "shape": shape,
                "operation": operation,
                "dimensions": dims,
                "formula": formula
            })

        else:
            return json.dumps({"error": f"Unknown mode: '{mode}'. Use 'expression' or 'geometry'."})

    except KeyError as e:
        return json.dumps({"error": f"Missing required dimension: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def count_letter(text: str, letter: str) -> str:
    """Count the number of occurrences of a single letter (case-insensitive)
    in a piece of text. Use this for questions like 'How many s's are in Mississippi?'"""
    if len(letter) != 1:
        return json.dumps({"error": "letter must be exactly one character"})
    count = text.lower().count(letter.lower())
    return json.dumps({
        "text": text,
        "letter": letter.lower(),
        "count": count
    })


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between units of measurement.
    Supported units — Temperature: fahrenheit, celsius, kelvin.
    Distance: miles, km, feet, meters, inches, cm.
    Weight: pounds, kg, grams, oz.
    Example: value=72, from_unit='fahrenheit', to_unit='celsius'."""
    f, t = from_unit.lower().strip(), to_unit.lower().strip()

    aliases = {"kilometres": "km", "kilometers": "km", "metres": "meters",
               "kilogram": "kg", "kilograms": "kg", "gram": "grams",
               "pound": "pounds", "ounce": "oz", "ounces": "oz",
               "inch": "inches", "foot": "feet", "centimeter": "cm",
               "centimeters": "cm", "centimetre": "cm", "centimetres": "cm"}
    f = aliases.get(f, f)
    t = aliases.get(t, t)

    TEMP = {"fahrenheit", "celsius", "kelvin"}
    DIST = {"miles", "km", "feet", "meters", "inches", "cm"}
    WEIGHT = {"pounds", "kg", "grams", "oz"}

    def group(unit):
        if unit in TEMP:   return "temperature"
        if unit in DIST:   return "distance"
        if unit in WEIGHT: return "weight"
        return None

    if group(f) is None:
        return json.dumps({"error": f"Unknown unit: '{from_unit}'"})
    if group(t) is None:
        return json.dumps({"error": f"Unknown unit: '{to_unit}'"})
    if group(f) != group(t):
        return json.dumps({"error": f"Cannot convert between {group(f)} and {group(t)}"})

    try:
        # temperature conversions
        if group(f) == "temperature":
            if f == t:
                result = value
            elif f == "fahrenheit" and t == "celsius":
                result = (value - 32) * 5 / 9
            elif f == "fahrenheit" and t == "kelvin":
                result = (value - 32) * 5 / 9 + 273.15
            elif f == "celsius" and t == "fahrenheit":
                result = value * 9 / 5 + 32
            elif f == "celsius" and t == "kelvin":
                result = value + 273.15
            elif f == "kelvin" and t == "celsius":
                result = value - 273.15
            elif f == "kelvin" and t == "fahrenheit":
                result = (value - 273.15) * 9 / 5 + 32
            else:
                return json.dumps({"error": f"Unsupported temperature conversion: {f} to {t}"})

        # distance conversions -> normalizes to meters first
        elif group(f) == "distance":
            to_meters = {"miles": 1609.344, "km": 1000, "feet": 0.3048,
                         "meters": 1, "inches": 0.0254, "cm": 0.01}
            meters = value * to_meters[f]
            result = meters / to_meters[t]

        # weight conversions -> normalizes to grams first
        elif group(f) == "weight":
            to_grams = {"pounds": 453.592, "kg": 1000, "grams": 1, "oz": 28.3495}
            grams = value * to_grams[f]
            result = grams / to_grams[t]

        else:
            return json.dumps({"error": f"Unhandled unit group for '{from_unit}'"})

        return json.dumps({
            "input_value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "result": round(result, 6)
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Build tool list and lookup map
tools = [get_weather, calculator, count_letter, unit_converter]
tool_map = {tool.name: tool for tool in tools}

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """

    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]

    print(f"User: {user_query}\n")

    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")

        # Call the LLM
        response = llm_with_tools.invoke(messages)

        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")

            # Add the assistant's response to messages
            messages.append(response)

            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]

                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")

                # Dispatch via tool_map
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"

                print(f"  Result: {result}")

                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))

            print()
            # Loop continues - LLM will see the tool results

        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content

    print("Max iterations reached\n")
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    print("="*60)
    print("TEST 1: Query requiring tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")

    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")

    print("\n" + "="*60)
    print("TEST 3: Multiple tool calls")
    print("="*60)
    run_agent("What's the weather in New York and London?")

    print("\n" + "="*60)
    print("TEST 4: Letter count - two calls in one turn")
    print("="*60)
    run_agent("Are there more i's than s's in the phrase 'Mississippi riverboats'?")

    print("\n" + "="*60)
    print("TEST 5: Letter count → calculator chaining")
    print("="*60)
    run_agent(
        "What is the sine of the difference between the number of i's and the "
        "number of s's in 'Mississippi riverboats'?"
    )

    print("\n" + "="*60)
    print("TEST 6: Unit converter standalone")
    print("="*60)
    run_agent("Convert 98.6 degrees Fahrenheit to Celsius and Kelvin.")

    print("\n" + "="*60)
    print("TEST 7: All four tools in one query")
    print("="*60)
    run_agent(
        "What's the weather in London? Count the number of 'o's in 'London'. "
        "Convert the London temperature from Fahrenheit to Celsius. "
        "What is the area of a circle whose radius equals the number of 'o's?"
    )

    print("\n" + "="*60)
    print("TEST 8: Deep sequential chain (approaches 5-iteration limit)")
    print("="*60)
    run_agent(
        "Count the 'e's in 'Tennessee'. Convert that many degrees Fahrenheit to "
        "Celsius. Compute the sine of the Celsius value. Use that sine value as the "
        "radius to find the area of a circle. Convert that area (treating it as "
        "miles) to kilometers. Show all intermediate values."
    )
