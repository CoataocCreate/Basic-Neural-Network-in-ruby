require 'matrix'

class NeuralNetwork
  attr_reader :input_size, :hidden_size, :output_size

  def initialize(input_size, hidden_size, output_size)
    @input_size = input_size
    @hidden_size = hidden_size
    @output_size = output_size

    @input_hidden_weights = Matrix.build(hidden_size, input_size) { rand * 0.2 - 0.1 }
    @hidden_output_weights = Matrix.build(output_size, hidden_size) { rand * 0.2 - 0.1 }
  end

  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end

  def sigmoid_derivative(x)
    x * (1.0 - x)
  end

  def forward(input)
    @hidden_layer_input = @input_hidden_weights * Matrix.column_vector(input)
    @hidden_layer = @hidden_layer_input.map { |x| sigmoid(x) }
    
    @output_layer_input = @hidden_output_weights * @hidden_layer
    @output_layer = @output_layer_input.map { |x| sigmoid(x) }
    
    @output_layer.to_a.flatten
  end

  def train(inputs, expected_outputs, epochs, learning_rate)
    inputs.each_with_index do |input, i|
      expected_output = Matrix.column_vector(expected_outputs[i])

      # Forward propagation
      forward(input)

      # Calculate errors
      output_error = expected_output - Matrix.column_vector(@output_layer)
      hidden_error = (@hidden_output_weights.transpose * output_error).map do |x|
        sigmoid_derivative(x)
      end

      # Update weights
      @hidden_output_weights += learning_rate * (output_error * @hidden_layer.transpose)
      @input_hidden_weights += learning_rate * (hidden_error * Matrix.column_vector(input).transpose)
    end
  end
end

# XOR problem
inputs = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]

expected_outputs = [
  [0],
  [1],
  [1],
  [0]
]

nn = NeuralNetwork.new(2, 4, 1)
nn.train(inputs, expected_outputs, 10000, 0.1)

inputs.each do |input|
  output = nn.forward(input)
  puts "Input: #{input.inspect} => Output: #{output.inspect}"
end
