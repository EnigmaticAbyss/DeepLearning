import pattern

def main():
    checker  = pattern.Checker(12,2)
    checker.draw()
    checker.show()
      
    circle  = pattern.Circle(400,100,(200,200))
    circle.draw()
    circle.show()
    
    
if __name__=="__main__":
    main()