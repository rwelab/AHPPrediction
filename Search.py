def lines_that_equal(line_to_match, fp):
    return [line for line in fp if line == line_to_match]

def lines_that_contain(string, fp):
    return [line for line in fp if string in line]

def lines_that_start_with(string, fp):
    return [line for line in fp if line.startswith(string)]

def lines_that_end_with(string, fp):
    return [line for line in fp if line.endswith(string)]
	
with open("file path", "r") as fp:
    for line in lines_that_contain("Porphyria", fp):
        print (line)
        
	
with open("file path", "r") as fp:
    for line in lines_that_contain("AHP", fp):
        print (line)