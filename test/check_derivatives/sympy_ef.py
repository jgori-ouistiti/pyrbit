from sympy import symbols, exp, log, diff

x, a, b, k, d = symbols("x a b k d")
variables = [a, b]


# q1
log_f = -a * (1 - b) ** k * d
print(str(diff(log_f, a)).replace("log", "numpy.log").replace("exp", "numpy.exp"))
print(str(diff(log_f, b)).replace("log", "numpy.log").replace("exp", "numpy.exp"))
dd = []
for row_variable in variables:
    row = []
    for column_variable in variables:
        row.append(
            str(diff(diff(log_f, column_variable), row_variable))
            .replace("log", "numpy.log")
            .replace("exp", "numpy.exp")
        )
    dd.append(row)

p1 = exp(log_f)
log_f = log(1 - exp(-a * (1 - b) ** k * d))
p0 = exp(log_f)

diff(p1 / p0, b)
exit()


# q0
print("==========================")
str(diff(log_f, a)).replace("log", "numpy.log").replace("exp", "numpy.exp")
str(diff(log_f, b)).replace("log", "numpy.log").replace("exp", "numpy.exp")
diff(log_f, b)


dd = []
for row_variable in variables:
    row = []
    for column_variable in variables:
        row.append(
            str(diff(diff(log_f, column_variable), row_variable))
            .replace("log", "numpy.log")
            .replace("exp", "numpy.exp")
        )
    dd.append(row)
exit()
