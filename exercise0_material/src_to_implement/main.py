import pattern
import generator

 
def main():
    checker  = pattern.Checker(12,2)
    checker.draw()
    checker.show()
      
    circle  = pattern.Circle(400,100,(300,200))
    circle.draw()
    circle.show()
    
    spectrum  = pattern.Spectrum(1000)
    spectrum.draw()
    spectrum.show()
    
    images = generator.ImageGenerator("exercise_data","./Labels.json",3,[100,100,3],False,False,True)    
    images.show()
   
    
if __name__=="__main__":
    main()
    
