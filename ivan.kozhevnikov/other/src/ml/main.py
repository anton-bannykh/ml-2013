import util
import regularization

input_data = zip(*util.read_input())
c = regularization.find_regularization_const(input_data)

result = regularization.build_svm_get_result(input_data, c)
print("The best result is when C is: " + str(c))
util.print_result(*result)