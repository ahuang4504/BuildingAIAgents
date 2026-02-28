"""
Manual Tool Calling Exercise
Students will see how tool calling works under the hood.
"""

import json
import math
import os
import numexpr  # pip install numexpr (in .venv)
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================
# PART 1: Define Your Tools
# ============================================

def get_weather(location: str) -> str:
    """Get the current weather for a location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


def calculator(mode: str, expression: str = None, shape: str = None,
               dimensions: str = None, operation: str = None) -> str:
    """Calculator with expression evaluation and geometric functions."""
    try:
        if mode == "expression":
            if expression is None:
                return json.dumps({"error": "expression is required in expression mode"})
            result = float(numexpr.evaluate(expression))
            return json.dumps({"result": result, "mode": "expression", "expression": expression})

        elif mode == "geometry":
            if not all([shape, dimensions, operation]):
                return json.dumps({"error": "shape, dimensions, and operation are required in geometry mode"})

            dims = json.loads(dimensions)  # parse JSON string input explicitly
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

            return json.dumps({  # format output with json.dumps explicitly
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


# ============================================
# PART 2: Describe Tools to the LLM
# ============================================

# This is the JSON schema that tells the LLM what tools exist
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name, e.g. San Francisco"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "A multi-purpose calculator supporting general expression evaluation and "
                "geometric property calculations (area, perimeter, volume, surface_area) "
                "for 2D and 3D shapes. Use mode='expression' for arithmetic like '2 * pi * 5', "
                "or mode='geometry' for shape calculations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["expression", "geometry"],
                        "description": "'expression' evaluates a math string; 'geometry' computes shape properties."
                    },
                    "expression": {
                        "type": "string",
                        "description": "Math expression string, e.g. '2**10 + sqrt(144)'. Required when mode='expression'."
                    },
                    "shape": {
                        "type": "string",
                        "enum": ["circle", "rectangle", "triangle", "square",
                                 "sphere", "cylinder", "cone", "cube", "rectangular_prism"],
                        "description": "Geometric shape. Required when mode='geometry'."
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["area", "perimeter", "volume", "surface_area"],
                        "description": "2D shapes: area/perimeter. 3D shapes: volume/surface_area. Required when mode='geometry'."
                    },
                    "dimensions": {
                        "type": "string",
                        "description": (
                            "JSON-encoded string of dimension values. Examples: "
                            "'{\"radius\": 5}' for circle/sphere, "
                            "'{\"width\": 4, \"height\": 6}' for rectangle/cylinder, "
                            "'{\"a\": 3, \"b\": 4, \"c\": 5}' for triangle, "
                            "'{\"side\": 7}' for square/cube, "
                            "'{\"radius\": 3, \"height\": 8}' for cone, "
                            "'{\"length\": 3, \"width\": 4, \"height\": 5}' for rectangular_prism. "
                            "Required when mode='geometry'."
                        )
                    }
                },
                "required": ["mode"]
            }
        }
    }
]


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Start conversation with user query
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,  # ← This tells the LLM what tools are available
            tool_choice="auto"  # Let the model decide whether to use tools
        )
        
        assistant_message = response.choices[0].message
        
        # Check if the LLM wants to call a tool
        if assistant_message.tool_calls:
            print(f"LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(assistant_message)
            
            # Execute each tool call
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # THIS IS THE MANUAL DISPATCH
                # In a real system, you'd use a dictionary lookup
                if function_name == "get_weather":
                    result = get_weather(**function_args)
                elif function_name == "calculator":
                    result = calculator(**function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": result
                })
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {assistant_message.content}\n")
            return assistant_message.content
    
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
    print("TEST 4: Expression evaluation")
    print("="*60)
    run_agent("What is 2 to the power of 10, plus the square root of 144?")

    print("\n" + "="*60)
    print("TEST 5: 2D geometry - circle area and perimeter")
    print("="*60)
    run_agent("What is the area and perimeter of a circle with radius 7?")

    print("\n" + "="*60)
    print("TEST 6: 3D geometry - cylinder volume and surface area")
    print("="*60)
    run_agent("Calculate the volume and surface area of a cylinder with radius 3 and height 10.")

    print("\n" + "="*60)
    print("TEST 7: Mixed tools - geometry + weather")
    print("="*60)
    run_agent("What's the weather in Tokyo, and what's the volume of a sphere with radius 4?")
