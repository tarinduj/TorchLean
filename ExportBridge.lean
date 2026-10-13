import NN.Runtime.PyTorch.Export.TorchExport

/-- `lake env lean --run ExportBridge.lean > export_torchlean_graph.py`: print the PyTorch capture script. -/
def main : IO Unit :=
  IO.print (Export.PyTorch.TorchExport.script {})
