import tkinter as tk
from tkinter import ttk, messagebox
import sounddevice as sd
import numpy as np
import soundfile as sf
import os, random, time, threading

RATE=44100
BLOCK=512
BASE=os.path.dirname(os.path.abspath(__file__))

class BoneVoice:
    def __init__(self, root):
        self.root=root
        self.running=False
        self.stream=None
        self.active=[]
        self.last_trigger=0
        self.samples=[]
        self.load()

        dev=sd.query_devices()
        ins=[f"{i}: {d['name']}" for i,d in enumerate(dev) if d['max_input_channels']>0]
        outs=[f"{i}: {d['name']}" for i,d in enumerate(dev) if d['max_output_channels']>0]

        ttk.Label(root,text="Microphone").grid(row=0,column=0,padx=10,pady=8,sticky="w")
        self.inbox=ttk.Combobox(root,values=ins,width=58,state="readonly")
        self.inbox.grid(row=0,column=1,padx=10,pady=8)
        if ins:self.inbox.current(0)

        ttk.Label(root,text="VB-CABLE Output").grid(row=1,column=0,padx=10,pady=8,sticky="w")
        self.outbox=ttk.Combobox(root,values=outs,width=58,state="readonly")
        self.outbox.grid(row=1,column=1,padx=10,pady=8)
        # Prefer CABLE Input as the output device (audio is fed into it).
        for i,x in enumerate(outs):
            if "CABLE Input" in x:
                self.outbox.current(i); break
        else:
            if outs:self.outbox.current(0)

        self.sens=tk.DoubleVar(value=1.3)
        self.rate=tk.DoubleVar(value=1.0)
        self.min_gap=tk.DoubleVar(value=0.065)

        ttk.Label(root,text="Sensitivity").grid(row=2,column=0,sticky="w",padx=10)
        ttk.Scale(root,from_=0.5,to=3,variable=self.sens).grid(row=2,column=1,sticky="ew",padx=10)
        ttk.Label(root,text="Clack density").grid(row=3,column=0,sticky="w",padx=10)
        ttk.Scale(root,from_=0.4,to=2.2,variable=self.rate).grid(row=3,column=1,sticky="ew",padx=10)

        self.meter=ttk.Progressbar(root,maximum=1,length=450)
        self.meter.grid(row=4,column=0,columnspan=2,padx=10,pady=12)
        self.status=ttk.Label(root,text="STOPPED")
        self.status.grid(row=5,column=0,columnspan=2)

        self.btn=ttk.Button(root,text="START",command=self.toggle)
        self.btn.grid(row=6,column=0,columnspan=2,pady=12)

        root.protocol("WM_DELETE_WINDOW",self.close)

    def load(self):
        d=os.path.join(BASE,"samples")
        for f in os.listdir(d):
            if f.lower().endswith(".wav"):
                x,sr=sf.read(os.path.join(d,f),dtype="float32")
                if x.ndim>1:x=x.mean(1)
                if sr!=RATE:
                    old=np.linspace(0,1,len(x),endpoint=False)
                    new=np.linspace(0,1,int(len(x)*RATE/sr),endpoint=False)
                    x=np.interp(new,old,x).astype("float32")
                self.samples.append(x)
        if not self.samples: raise RuntimeError("No samples found")

    def trigger(self,strength):
        x=random.choice(self.samples).copy()
        # Random amplitude and tiny time/pitch variation.
        x*=np.clip(.25+strength*.95,.2,1.2)
        factor=random.uniform(.88,1.12)
        idx=np.linspace(0,len(x)-1,max(1,int(len(x)/factor)))
        x=np.interp(idx,np.arange(len(x)),x).astype("float32")
        self.active.append(x)

    def callback(self,indata,outdata,frames,t,status):
        # Never copy the original mic signal to output.
        mono=indata[:,0]
        rms=float(np.sqrt(np.mean(mono*mono))+1e-12)
        level=np.clip(rms*self.sens.get()*8,0,1)
        now=time.monotonic()

        # Voice activity -> bone events. Stronger speech creates louder/dense clacks.
        if rms>0.025 and now-self.last_trigger > max(.045,.18/(.35+self.rate.get()*max(level,.1))):
            self.trigger(min(1,rms*8*self.sens.get()))
            self.last_trigger=now

        out=np.zeros(frames,dtype=np.float32)
        new=[]
        for x in self.active:
            n=min(frames,len(x))
            out[:n]+=x[:n]
            if n<len(x): new.append(x[n:])
        self.active=new
        out=np.clip(out,-.9,.9)
        outdata[:,0]=out
        if outdata.shape[1]>1: outdata[:,1]=out

        self.root.after_idle(lambda v=float(level): self.meter.configure(value=v))

    def toggle(self):
        if self.running:
            self.stop()
            return
        try:
            i=int(self.inbox.get().split(":",1)[0])
            o=int(self.outbox.get().split(":",1)[0])
            self.stream=sd.Stream(device=(i,o),samplerate=RATE,blocksize=BLOCK,
                                  channels=(1,2),dtype="float32",callback=self.callback)
            self.stream.start()
            self.running=True
            self.btn.configure(text="STOP")
            self.status.configure(text="RUNNING — Discord should use CABLE Output")
        except Exception as e:
            messagebox.showerror("Start failed",str(e))

    def stop(self):
        self.running=False
        if self.stream:
            try:self.stream.stop();self.stream.close()
            except:pass
        self.stream=None
        self.active=[]
        self.btn.configure(text="START")
        self.status.configure(text="STOPPED")

    def close(self):
        self.stop();self.root.destroy()

root=tk.Tk()
root.title("Bone Voice")
root.resizable(False,False)
BoneVoice(root)
root.mainloop()
