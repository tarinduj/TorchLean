import NN.Runtime.PyTorch.Export.TorchExport

/-- `lake exe export_bridge > export_torchlean_graph.py`: the PyTorch → TorchLean IR capture script. -/
def main : IO Unit :=
  IO.print (Export.PyTorch.TorchExport.generateGraphBridgeScript {})
