def Propagate2Pyramid_tf(pupil,phaseMap,pyrMask,fovInPixel,nPxPup,Samp,modulation,alpha,rooftop,r,o,nTheta,ModPhasor,OL1):
    nTheta = torch.tensor(nTheta,dtype=tf.int16)
    PyrQ  = torch.zeros((fovInPixel,fovInPixel))
    I4Q4 =  torch.zeros((fovInPixel,fovInPixel))
    pyrPupil = pupil*torch.exp(1j*phaseMap)
    subscale = 1/(2*Samp)
    sx = tf.math.round(fovInPixel*subscale) 
    npv = (fovInPixel)-sx)/2
    PyrQ = tf.pad(pyrPupil, [[npv, npv], [npv, npv]], "CONSTANT"),tf.complex64)
    nTheta_f = tf.cast(nTheta,tf.float32)
    
    for kTheta in range(nTheta):               
        #APhase-->ModMirror-->OpticLayer1--->OpticLayer2-->Pyramid-->Sensor
        buf = PyrQ*ModPhasor[:,:,kTheta]  
        buf = tf.signal.fft2d(buf)*OL1    
        buf = tf.signal.fft2d(buf)       
        I4Q4 = I4Q4 + tf.math.abs(buf)**2   
    I4Q = I4Q4/nTheta_f    
    I4Q = tf.squeeze(tf.image.resize(tf.expand_dims(I4Q,2),[sx,sx])*2*Samp)
    return(I4Q)   