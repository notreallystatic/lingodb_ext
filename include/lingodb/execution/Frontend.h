#ifndef LINGODB_EXECUTION_FRONTEND_H
#define LINGODB_EXECUTION_FRONTEND_H
#include "Error.h"
#include "lingodb/catalog/Catalog.h"
#include "lingodb/compiler/Dialect/TupleStream/ColumnManager.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

#include <memory>

namespace mlir {
// class ModuleOp;
class MLIRContext;
} // namespace mlir
namespace lingodb::execution {
class Frontend {
   protected:
   catalog::Catalog* catalog;
   Error error;

   std::unordered_map<std::string, double> timing;

   public:
   catalog::Catalog* getCatalog() const {
      return catalog;
   }
   void setCatalog(catalog::Catalog* catalog) {
      Frontend::catalog = catalog;
   }
   const std::unordered_map<std::string, double>& getTiming() const {
      return timing;
   }
   Error& getError() { return error; }
   virtual void loadFromFile(std::string fileName) = 0;
   virtual void loadFromString(std::string data) = 0;
   virtual void loadFromGlobalContext() = 0;
   virtual bool isParallelismAllowed() { return true; }
   virtual mlir::ModuleOp* getModule() = 0;
   virtual ~Frontend() {}
};
std::unique_ptr<Frontend> createMLIRFrontend();
std::unique_ptr<Frontend> createSQLFrontend();
void initializeContext(mlir::MLIRContext& context);

class MLIRContainer {
   private:
   MLIRContainer();
   bool initialized = false;

   public:
   mlir::MLIRContext context;
   mlir::OpBuilder builder;
   mlir::OwningOpRef<mlir::ModuleOp> moduleOp;
   mlir::OpPrintingFlags flags;

   mlir::Block* predBlock;

   static MLIRContainer& getInstance() {
      static MLIRContainer instance;
      if (!instance.initialized) {
         instance.initialize();
      }
      return instance;
   }

   void initialize();
   // void createMainFuncBlock();
   void print();
   void printInfo();

   mlir::MLIRContext& getContext() { return context; }
   mlir::MLIRContext* getContextPtr() { return &context; }
   mlir::ModuleOp* getModuleOpPtr() { return moduleOp.operator->(); }
   mlir::ModuleOp getModuleOp() { return moduleOp.get(); }
   // mlir::ModuleOp* getModuleOpPtr() { return moduleOp; }
   mlir::OpBuilder& getBuilder() { return builder; }
   mlir::OpPrintingFlags& getFlags() { return flags; }
   void setPredBlock(mlir::Block* block) { predBlock = block; }
   mlir::Block* getPredBlock() { return predBlock; }
};

} //namespace lingodb::execution

#endif //LINGODB_EXECUTION_FRONTEND_H
